"""
Deteccion de presencia y DISTANCIA de una persona parada o que se mueve despacio, a partir del CIR con clutter removal.

Por que asi (y no un umbral de |residuo|^2 por notificacion con M-de-N):
  * Una persona parada sigue respirando / balanceandose. A 8 GHz (lambda ~ 3.8 cm) unos mm de movimiento del pecho giran la
    fase de su eco ~1 rad: su eco VARIA en el tiempo aunque no ande. El clutter estatico (paredes, muebles) no varia.
  * La VARIANZA TEMPORAL de la senal compleja en cada tap, integrada durante varios segundos, junta muchisima mas energia que
    3 aciertos sueltos de 4, y no necesita un fondo perfecto: la media (donde vive el clutter) se resta sola. Una persona
    quieta, que un fondo adaptativo acaba absorbiendo, sigue siendo visible.
  * El score no es una Pfa teorica sino un cociente respecto a un suelo MEDIDO (local en distancia + aprendido con la sala
    vacia), asi que absorbe el ruido real (jitter del clutter, deriva) que el modelo gaussiano ignoraba.

Pipeline (cada notificacion con el filtro de clutter ya listo):
  1. Dos ramas, cada una con media y varianza exponenciales del residuo complejo por tap y RX (O(1), sin buffers):
       QUIETA     : antes de la varianza se promedian ~4 notificaciones (paso bajo). La respiracion (0.2-0.4 Hz) pasa intacta y
                    el ruido blanco baja ~7 dB: detecta personas paradas mucho mas debiles. Ventana larga (PRESENCE_TAU_S).
       MOVIMIENTO : varianza de la senal sin filtrar y ventana corta (PRESENCE_TAU_FAST_S): quien camina despacio gira la fase
                    muy rapido (10 Hz a 0.2 m/s) y el paso bajo se lo comeria; ademas sigue mejor la posicion.
  2. Cada PRESENCE_UPDATE_FRAMES notificaciones: score[t] = max(rama QUIETA, rama MOVIMIENTO); cada una = sum_RX(varianza)/suelo,
       suelo = max(mediana local en distancia (OS-CFAR), suelo aprendido sin presencia).
  3. Picos del score -> seguimiento: puerta alrededor de la posicion anterior, confirmacion, retencion y reenganche.
  4. Distancia = centroide del pico (sub-tap) suavizado.

La salida (historial) alimenta la grafica en vivo (CIR arriba, distancia abajo), el CSV y la figura final de la sesion.
"""
import warnings
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config as cfg
from util import output


def local_floor(p, i0, train, guard):
    """Suelo local de un perfil p (T,): mediana de las celdas vecinas sin las de guarda. Ignora las celdas < i0 (zona ciega)."""
    n = len(p)
    k = train + guard
    pad = np.full(n + 2 * k, np.nan)
    pad[k:k + n] = p
    pad[k:k + i0] = np.nan
    win = sliding_window_view(pad, 2 * k + 1).copy()          # (n, 2k+1): [train | guard | CUT | guard | train]
    win[:, train:train + 2 * guard + 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fl = np.nanmedian(win, axis=1)
    bad = ~np.isfinite(fl)
    if bad.any():
        fl[bad] = np.nanmedian(p[i0:]) if n > i0 else 1.0
    return fl


def find_peaks(score, i0, thr, nms=3, kmax=3):
    """Maximos locales del score >= thr (a partir del tap i0), los mas fuertes primero y separados mas de `nms` taps."""
    if len(score) < 3:
        return []
    c = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:]) & (score[1:-1] >= thr)
    idx = np.flatnonzero(c) + 1
    idx = idx[idx >= i0]
    keep = []
    for i in idx[np.argsort(-score[idx])]:
        if all(abs(int(i) - j) > nms for j in keep):
            keep.append(int(i))
        if len(keep) >= kmax:
            break
    return keep


def centroid(score, i, i0, half=2):
    """Posicion sub-tap del pico: centroide del exceso de score (potencia) en +-half taps."""
    lo, hi = max(i0, i - half), min(len(score), i + half + 1)
    w = np.maximum(score[lo:hi] - 1.0, 0.0)
    s = w.sum()
    return float((np.arange(lo, hi) * w).sum() / s) if s > 0 else float(i)


class _Branch:
    """Varianza temporal exponencial por (rx, tap) con suelo local + suelo aprendido. `lp_name`: paso bajo previo (en notificaciones)."""

    def __init__(self, tau_name, lp_name=None):
        self.tau_name, self.lp_name = tau_name, lp_name
        self.reset()

    def reset(self):
        self.m = self.v = self.xl = None
        self.n = 0
        self.lt = None              # suelo aprendido por tap (sala vacia)
        self.lt_n = 0

    @property
    def lam(self):
        return min(1.0, 1.0 / (getattr(cfg, self.tau_name) * 1000.0 / cfg.RANGING_INTERVAL_MS))

    @property
    def n_ready(self):
        return int(2.0 / self.lam)

    def update(self, x):
        self.n += 1
        lp = getattr(cfg, self.lp_name) if self.lp_name else 1
        if lp > 1:                                                      # paso bajo: ~media movil de `lp` notificaciones
            self.xl = x.astype(complex) if self.xl is None else self.xl + max(1.0 / lp, 1.0 / self.n) * (x - self.xl)
            x = self.xl
        if self.m is None:
            self.m, self.v = x.astype(complex), np.zeros(x.shape)
            return
        a = max(self.lam, 1.0 / self.n)                                 # al principio acumuladas; luego exponenciales
        d = x - self.m
        self.m = self.m + a * d
        self.v = (1.0 - a) * (self.v + a * (d.real ** 2 + d.imag ** 2))

    def score(self, i0, learn):
        vc = self.v.sum(axis=0)                                         # varianza total por tap (suma de RX)
        loc = local_floor(vc, i0, cfg.PRESENCE_CFAR_TRAIN, cfg.PRESENCE_CFAR_GUARD)
        if self.lt is None or self.lt.shape != vc.shape:
            self.lt, self.lt_n = loc.copy(), 0
        sc = vc / np.maximum(np.maximum(loc, self.lt), 1e-12)
        sc[:i0] = 0.0
        if learn:                                                       # el suelo solo se aprende sin presencia ni candidato
            self.lt_n += 1
            near = np.arange(len(vc)) * cfg.METERS_PER_SAMPLE < cfg.PRESENCE_NEAR_M     # cerca del radar el jitter del clutter es enorme
            target = np.minimum(vc, np.where(near, cfg.PRESENCE_LEARN_MAX_NEAR, cfg.PRESENCE_LEARN_MAX) * loc)
            if self.lt_n <= cfg.PRESENCE_LEARN_CAL:                     # calibracion inicial: media acumulada (sala vacia)
                self.lt += (target - self.lt) / self.lt_n
            else:                                                       # luego: lento y solo en taps que parecen ruido
                self.lt += cfg.PRESENCE_LEARN_RATE * (sc < cfg.PRESENCE_LEARN_BELOW) * (target - self.lt)
        return sc


class PresenceDetector:
    def __init__(self):
        # historial: un elemento por evaluacion (se conserva al hacer reset(), p.ej. al capturar el fondo)
        self.t, self.idx_h = [], []
        self.cir_db, self.score_db = [], []             # columnas (T,) para los mapas distancia-tiempo
        self.present_h, self.score_h, self.raw_h, self.trk_h = [], [], [], []
        self.reset()

    def reset(self):
        """Reinicia los estimadores y el seguimiento (no el historial)."""
        self.quiet = _Branch("PRESENCE_TAU_S", "PRESENCE_LP_FRAMES")
        self.move = _Branch("PRESENCE_TAU_FAST_S")
        self.n = 0                  # notificaciones procesadas desde el reset
        self.k = 0
        self.p = None               # potencia suavizada por (rx, tap) (para dibujar el CIR)
        self.shape = None
        self.present = False
        self.cand_since = None      # inicio del candidato (aun sin confirmar)
        self.alt_since = None       # inicio de un pico fuerte fuera de la puerta
        self.last_meas_t = None
        self.track = None           # tap suavizado de la persona
        self.score_peak = 0.0
        self._vhist = deque()       # (t, varianza absoluta rama quieta) para detectar que una persona se apaga
        self._vprior = None
        self._fade_since = None
        self._hist = deque()        # (t, score rama quieta) de los ultimos segundos, para detectar fantasmas en decaimiento

    @property
    def warmup_frac(self):
        """0..1: avance del calentamiento (hasta que los estimadores se asientan no hay deteccion)."""
        need = max(self.quiet.n_ready, self.move.n_ready)
        return min(1.0, self.n / need) if need else 1.0

    # ---------------------------------------------------------------------------------------------------------------
    def update(self, clean, t, idx=-1):
        """Procesa una notificacion (clean: n_rx x n_taps complejo, ya sin clutter). Devuelve una fila al evaluar, si no None."""
        off = cfg.SAMPLE_OFFSET
        T = min(clean.shape[1] - off, int(cfg.PRESENCE_MAX_M / cfg.METERS_PER_SAMPLE) + 1)
        if T < 12:
            return None
        x = np.asarray(clean)[:, off:off + T]
        if self.shape != x.shape:
            self.reset()
            self.shape = x.shape
            self.p = np.abs(x) ** 2
        self.n += 1
        self.quiet.update(x)
        self.move.update(x)
        self.p = self.p + max(cfg.PRESENCE_CIR_SMOOTH, 1.0 / self.n) * (np.abs(x) ** 2 - self.p)
        if self.n < max(self.quiet.n_ready, self.move.n_ready):         # estimadores todavia sin asentar
            return None
        self.k += 1
        if self.k % max(1, cfg.PRESENCE_UPDATE_FRAMES):
            return None
        return self._evaluate(t, idx, T)

    # ---------------------------------------------------------------------------------------------------------------
    def _evaluate(self, t, idx, T):
        mps = cfg.METERS_PER_SAMPLE
        i0 = int(min(round(cfg.PRESENCE_MIN_M / mps), T - 8))
        learn = (not self.present) and self.cand_since is None
        sq, sm = self.quiet.score(i0, learn), self.move.score(i0, learn)
        score = np.maximum(sq, sm)
        self.score_peak = float(score[i0:].max())

        thr_on, thr_off = cfg.PRESENCE_THR_ON, cfg.PRESENCE_THR_OFF

        def peaks(thr):                                                 # (tap, score, perfil de origen); movimiento primero
            return [(i, float(s[i]), s) for s in (sm, sq) for i in find_peaks(s, i0, thr)]

        # Fantasma: tras irse una persona, la varianza de la rama quieta tarda varios tau en decaer y puede ser mas fuerte que el
        # eco real de quien acaba de llegar a otro sitio. Un fantasma se reconoce porque su exceso de score CAE; un objetivo real
        # se mantiene o sube. Se descartan los picos de la rama quieta cuyo exceso es < DECAY_RATIO del de hace DECAY_LAG_S.
        prior = None
        for tp, sp in self._hist:
            if t - tp >= cfg.PRESENCE_DECAY_LAG_S:
                prior = sp
            else:
                break
        self._hist.append((t, sq))
        vq = self.quiet.v.sum(axis=0)                                   # varianza absoluta por tap (suma de RX)
        self._vprior = None
        for tp, vp in self._vhist:
            if t - tp >= cfg.PRESENCE_DECAY_LAG_S:
                self._vprior = vp
            else:
                break
        self._vhist.append((t, vq.copy()))
        while len(self._vhist) > 1 and t - self._vhist[0][0] > cfg.PRESENCE_DECAY_LAG_S + 2.0:
            self._vhist.popleft()
        fading = self._fading(t, vq)
        while len(self._hist) > 1 and t - self._hist[0][0] > cfg.PRESENCE_DECAY_LAG_S + 2.0:
            self._hist.popleft()

        def abs_fading(tap):                                            # varianza absoluta de la rama quieta cayendo en este tap
            if self._vprior is None:
                return False
            lo, hi = max(0, tap - 2), tap + 3
            prev = float(self._vprior[lo:hi].max())
            return prev > 0 and float(vq[lo:hi].max()) < cfg.PRESENCE_FADE_RATIO * prev

        def decaying(c):
            if c[2] is sq and abs_fading(c[0]):                         # fantasma de alguien que se ha ido (o de un eco que se apaga)
                return True
            if prior is None or c[2] is not sq:
                return False
            lo, hi = max(0, c[0] - 1), c[0] + 2
            ex_prev = float(prior[lo:hi].max()) - 1.0
            return ex_prev > 0.3 and (float(sq[lo:hi].max()) - 1.0) < cfg.PRESENCE_DECAY_RATIO * ex_prev

        strong = [c for c in peaks(thr_on) if not decaying(c)]
        pk = [c for c in peaks(thr_off) if not decaying(c)] if self.present else strong

        meas, jump = None, False
        if self.track is not None:                                      # siguiendo: pico dentro de la puerta
            gate = (cfg.PRESENCE_GATE_M + cfg.PRESENCE_GATE_GROW * max(0.0, t - self.last_meas_t)) / mps
            inside = [c for c in pk if abs(c[0] - self.track) <= gate]
            if inside:
                # Quien se aleja deja un "fantasma" en la rama quieta (su varianza tarda varios tau en decaer) que puede
                # ser mas fuerte que su eco real. La rama de movimiento solo recuerda ~tau_fast, asi que manda dentro de la puerta.
                mv = [c for c in inside if c[2] is sm]
                meas = max(mv or inside, key=lambda c: c[1])
                self.alt_since = None
            elif strong:                                                # pico fuerte fuera de la puerta: reenganche
                if self.alt_since is None:
                    self.alt_since = t
                elif t - self.alt_since >= (cfg.PRESENCE_REACQ_FADE_S if fading else cfg.PRESENCE_REACQ_S):
                    meas, jump, self.alt_since = max(strong, key=lambda c: c[1]), True, None
            else:
                self.alt_since = None
        elif strong:
            meas = max(strong, key=lambda c: c[1])

        raw_tap = float("nan")
        if meas is not None:
            raw_tap = centroid(meas[2], meas[0], i0)
            self.last_meas_t = t
            if self.track is None or jump:
                self.track = raw_tap
            else:
                self.track += cfg.PRESENCE_POS_ALPHA * (raw_tap - self.track)
            if not self.present:
                if self.cand_since is None:
                    self.cand_since = t
                if t - self.cand_since >= cfg.PRESENCE_CONFIRM_S:
                    self.present = True
                    output("[PRESENCIA] DETECTADA a %.2f m (score %.1f, umbral %.1f)" % (self.track * mps, meas[1], thr_on))
        else:
            if not self.present:
                self.cand_since, self.track = None, None
            else:
                # Retencion: tras perder el eco se sigue PRESENTE PRESENCE_HOLD_S (evita parpadeo con gente quieta y debil). Pero si el
                # score alrededor de la ultima posicion se esta APAGANDO (cae por debajo de FADE_RATIO de su exceso de hace
                # DECAY_LAG_S) es que la persona se ha ido: se libera antes (PRESENCE_HOLD_FADE_S).
                lost = t - self.last_meas_t
                hold = cfg.PRESENCE_HOLD_FADE_S if fading else cfg.PRESENCE_HOLD_S
                if lost > hold:
                    self.present, self.track, self.cand_since = False, None, None
                    output("[PRESENCIA] AUSENTE")

        raw_m = raw_tap * mps
        trk_m = self.track * mps if (self.present and self.track is not None) else float("nan")

        pr = self.p.mean(axis=0)                                        # CIR sin clutter (potencia suavizada, media de RX)
        ref = max(float(np.median(pr[i0:])), 1e-9)
        self.t.append(t)
        self.idx_h.append(idx)
        self.cir_db.append((10.0 * np.log10(np.maximum(pr, 1e-6 * ref) / ref)).astype(np.float32))
        self.score_db.append((10.0 * np.log10(np.maximum(score, 1e-3))).astype(np.float32))
        self.present_h.append(int(self.present))
        self.score_h.append(self.score_peak)
        self.raw_h.append(raw_m)
        self.trk_h.append(trk_m)
        return (t, idx, int(self.present), self.score_peak, raw_m, trk_m)

    def _fading(self, t, vq):
        """True si la VARIANZA ABSOLUTA de la rama quieta alrededor de la ultima posicion se esta apagando (la persona se ha ido).

        No se usa el score: es una razon respecto al suelo local, y al irse alguien la varianza de toda la zona cae a la vez, asi que
        el score puede quedarse casi constante durante 10 s mientras la senal real ya ha desaparecido. Una persona que sigue ahi
        mantiene (con fluctuaciones) su varianza absoluta; un fantasma la ve caer ~0.5x cada DECAY_LAG_S.
        La condicion debe mantenerse PRESENCE_FADE_CONFIRM_S para no reaccionar a una fluctuacion puntual."""
        if self.track is None or self._vprior is None:
            self._fade_since = None
            return False
        c = int(round(self.track))
        lo, hi = max(0, c - 2), c + 3
        prev = float(self._vprior[lo:hi].max())
        now = float(vq[lo:hi].max())
        if prev > 0 and now < cfg.PRESENCE_FADE_RATIO * prev:
            if self._fade_since is None:
                self._fade_since = t
            return (t - self._fade_since) >= cfg.PRESENCE_FADE_CONFIRM_S
        self._fade_since = None
        return False

    # ---------------------------------------------------------------------------------------------------------------
    def history(self, last=None):
        """Historial como arrays numpy (las ultimas `last` evaluaciones, o todo)."""
        s = slice(None) if last is None else slice(-last, None)
        cir, sc = self.cir_db[s], self.score_db[s]
        nt = len(cir[0]) if cir else 0                  # (0, 0) mientras no hay datos
        return {
            "t": np.array(self.t[s], dtype=float),
            "idx": np.array(self.idx_h[s], dtype=int),
            "cir_db": np.array(cir, dtype=np.float32).reshape(len(cir), nt),
            "score_db": np.array(sc, dtype=np.float32).reshape(len(sc), nt),
            "present": np.array(self.present_h[s], dtype=int),
            "score": np.array(self.score_h[s], dtype=float),
            "raw_m": np.array(self.raw_h[s], dtype=float),
            "trk_m": np.array(self.trk_h[s], dtype=float),
        }


det = PresenceDetector()        # instancia global usada por la captura en vivo


def reset_presence():
    """Reinicia el detector (tecla 'n', fin de la captura de fondo...). Llamar con live_lock tomado."""
    det.reset()


def push_frame(clean, filters, cir_idx, t):
    """Procesa una notificacion (llamar con live_lock tomado). `filters` se mantiene por compatibilidad. Devuelve la fila si hay evaluacion."""
    return det.update(clean, t, cir_idx)

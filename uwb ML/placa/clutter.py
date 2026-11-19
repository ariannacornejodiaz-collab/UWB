"""
Clutter removal + deteccion de eco + media (solo numpy, sin puerto serie ni graficos):
  ClutterFilter  filtro adaptativo por RX (alineado complejo, resta, referencia compuerta)
  fondo          captura de escena vacia (tecla 'b')
  detect_echoes  OS-CFAR sobre el perfil medio, tap sub-muestra, SNR y distancia
  compute_average  media de la ventana (abs o coherente)
"""
import os

import numpy as np

import config as cfg
import state as st
from presence_detect import reset_presence
from util import output


class ClutterFilter:
    """
    Clutter removal adaptativo sobre el CIR complejo (I+Q), una instancia por RX.
      1. Alineado complejo (ganancia + fase) de cada notificacion contra la referencia, por minimos cuadrados
         sobre los taps de clutter: elimina la deriva de fase/ganancia entre notificaciones, que es lo que
         impide que el clutter fuerte se cancele con una simple resta.
      2. Resta:  residuo = CIR_alineado - referencia.
      3. Referencia con actualizacion COMPUERTA (gated): solo se adapta en los taps sin eco. Un tap tiene eco si
         |residuo| > K*sigma (fuerte) o si el residuo SUAVIZADO es persistente (eco quieto por debajo del ruido).
         En esos taps la referencia casi se congela, asi un objetivo quieto NO se absorbe como clutter.
      4. sigma por tap (ruido del residuo) estimada solo en taps sin eco -> umbral de puerta.
    Referencia inicial: fondo de escena vacia (set_background) o, si no hay, media de las primeras `warmup` notificaciones.
    `ready` indica que la referencia ya es fiable (solo entonces las notificaciones entran en la media).
    """
    def __init__(self, warmup=None, alpha=cfg.CLUTTER_ALPHA, leak=cfg.CLUTTER_LEAK, k_gate=cfg.GATE_K, k_smooth=cfg.GATE_KS,
                 beta=cfg.SMOOTH_BETA, sigma_rate=cfg.SIGMA_RATE):
        self.warmup = cfg.WARMUP_FRAMES if warmup is None else warmup
        self.alpha = alpha
        self.leak = leak
        self.k = k_gate
        self.ks = k_smooth
        self.beta = beta
        self.sm_gain = np.sqrt(beta / (2.0 - beta))   # std del residuo suavizado relativa a sigma
        self.sigma_rate = sigma_rate
        self._bg = None
        self._clear()

    def _clear(self):
        self.ref = None
        self.sigma = None
        self.mask = None            # True = tap de clutter (se usa para alinear)
        self.pw = None
        self.sm = None              # residuo complejo suavizado (EMA)
        self.sigma0 = None          # sigma al terminar la calibracion (la banda SIGMA_BAND se mide respecto a ella)
        self.n = 0
        self.ready = False

    def set_background(self, ref, sigma):
        self._bg = (np.array(ref, dtype=complex), np.array(sigma, dtype=float))
        self._apply_bg()

    def _apply_bg(self):
        ref, sigma = self._bg
        self.ref = ref.copy()
        self.sigma = np.maximum(sigma, cfg.SIGMA_MIN)
        self.sigma0 = self.sigma.copy()
        self.mask = np.ones(ref.shape, dtype=bool)
        self.n = self.warmup
        self.ready = True

    def reset(self):
        if self._bg is not None:
            self._apply_bg()
        else:
            self._clear()

    def _align(self, cur):
        m = self.mask
        if m is None or m.sum() < 4:
            m = np.ones(cur.shape, dtype=bool)
        r = self.ref[m]
        den = np.vdot(r, r).real
        if den < 1e-9:
            return cur
        g = np.vdot(r, cur[m]) / den                  # cur ~= g * ref
        if not (0.5 < abs(g) < 2.0):
            return cur
        return cur / g

    def process(self, complex_cir):
        cur = np.asarray(complex_cir, dtype=complex)
        if self.ref is None or self.ref.shape != cur.shape:
            self._clear()
            self.ref = cur.copy()
            self.mask = np.ones(cur.shape, dtype=bool)
            self.pw = np.zeros(cur.shape)
            self.sigma = np.full(cur.shape, cfg.SIGMA_MIN)
            self.n = 1
            return np.zeros_like(cur)

        aligned = self._align(cur)
        resid = aligned - self.ref

        if not self.ready:                            # calentamiento: media acumulada + varianza del residuo
            self.n += 1
            self.pw += (np.abs(resid) ** 2 - self.pw) / (self.n - 1)
            self.ref = self.ref + (aligned - self.ref) / self.n
            if self.n >= self.warmup:
                self.sigma = np.maximum(np.sqrt(self.pw), cfg.SIGMA_MIN)
                self.sigma0 = self.sigma.copy()
                self.ready = True
            return resid

        if self.sm is None or self.sm.shape != resid.shape:
            self.sm = np.zeros_like(resid)
        self.sm = self.sm + self.beta * (resid - self.sm)
        echo = (np.abs(resid) > self.k * self.sigma) | \
               (np.abs(self.sm) > self.ks * self.sm_gain * self.sigma)    # eco instantaneo o persistente
        upd = np.where(echo, self.leak, 1.0 - self.alpha)
        self.ref = self.ref + upd * (aligned - self.ref)
        s2 = self.sigma ** 2
        s2 = np.where(echo, s2, (1.0 - self.sigma_rate) * s2 + self.sigma_rate * np.abs(resid) ** 2)
        self.sigma = np.clip(np.sqrt(s2), cfg.SIGMA_BAND[0] * self.sigma0, cfg.SIGMA_BAND[1] * self.sigma0)
        self.mask = ~echo
        return resid


def get_filter(i, n_taps):
    f = st.clutter_filters.get(i)
    if f is None:
        f = ClutterFilter()
        if st.bg_data is not None and i < len(st.bg_data["ref"]) and len(st.bg_data["ref"][i]) == n_taps:
            f.set_background(st.bg_data["ref"][i], st.bg_data["sigma"][i])
        st.clutter_filters[i] = f
    return f


def start_background_capture():
    with st.live_lock:
        st.bg_capture["active"] = True
        st.bg_capture["filters"] = {}
        st.bg_capture["n"] = 0


def feed_background(cur):
    """Acumula notificaciones de escena vacia (llamar con live_lock tomado). Al terminar activa el fondo."""
    fl = st.bg_capture["filters"]
    for i in range(len(cur)):
        fl.setdefault(i, ClutterFilter(warmup=cfg.BG_FRAMES)).process(cur[i])
    st.bg_capture["n"] += 1
    if all(f.ready for f in fl.values()):
        ids = sorted(fl)
        st.bg_data = {"ref": np.array([fl[i].ref for i in ids]), "sigma": np.array([fl[i].sigma for i in ids])}
        try:
            np.savez(os.path.join(st.cir_dir, "background.npz"), **st.bg_data)
        except Exception:
            pass
        st.bg_capture["active"] = False
        st.clutter_filters.clear()
        _reset_unlocked()
        output("[BG] Fondo capturado (%d notificaciones) y activado. Pulsa 'n' para medir el escenario." % cfg.BG_FRAMES)


def detect_echoes(profile):
    """
    Deteccion de eco sobre el perfil medio (OS-CFAR): el suelo de ruido local es la mediana de las celdas vecinas
    (sin las de guarda); el umbral depende del modo de media (abs: mediana+K*MAD, coherent: Rayleigh con Pfa fija).
    Devuelve (lista de ecos [{tap, amp, snr_db}], umbral por tap). El tap tiene precision sub-muestra (parabola).
    """
    p = np.asarray(profile, dtype=float)
    n = len(p)
    thr = np.zeros(n)
    noise = np.zeros(n)
    for i in range(n):
        lo = max(0, i - cfg.CFAR_GUARD - cfg.CFAR_TRAIN)
        hi = min(n, i + cfg.CFAR_GUARD + 1 + cfg.CFAR_TRAIN)
        cells = np.concatenate([p[lo:max(0, i - cfg.CFAR_GUARD)], p[min(n, i + cfg.CFAR_GUARD + 1):hi]])
        if len(cells) < 4:
            cells = p
        med = np.median(cells)
        mad = 1.4826 * np.median(np.abs(cells - med))
        noise[i] = med
        if cfg.AVG_MODE == "coherent":                      # ruido Rayleigh: umbral multiplicativo (Pfa constante)
            thr[i] = max(cfg.CFAR_RAYLEIGH_FACTOR * med, cfg.CFAR_MIN_RATIO * med)
        else:                                           # media de |CIR|: ruido casi gaussiano
            thr[i] = max(med + cfg.CFAR_K * max(mad, 0.05 * med), cfg.CFAR_MIN_RATIO * med)

    echoes = []
    for i in range(max(cfg.ECHO_MIN_TAP, 0), n):
        l = p[i - 1] if i > 0 else -np.inf
        r = p[i + 1] if i < n - 1 else -np.inf
        if p[i] > thr[i] and p[i] >= l and p[i] >= r:
            tap, amp = float(i), float(p[i])
            if 0 < i < n - 1:
                den = p[i - 1] - 2 * p[i] + p[i + 1]
                if den < 0:
                    delta = float(np.clip(0.5 * (p[i - 1] - p[i + 1]) / den, -0.5, 0.5))
                    tap = i + delta
                    amp = float(p[i] - 0.25 * (p[i - 1] - p[i + 1]) * delta)
            snr = 20 * np.log10(max(amp, 1e-9) / max(noise[i], 1e-9))
            echoes.append({"tap": tap, "amp": amp, "snr_db": float(snr),
                           "sample": tap + cfg.SAMPLE_OFFSET,                 # indice en el CIR original
                           "dist_m": tap * cfg.METERS_PER_SAMPLE})            # distancia corregida con el offset
    echoes.sort(key=lambda e: -e["amp"])
    return echoes[:cfg.MAX_ECHOES], thr


def detect_all(avg):
    res = [detect_echoes(row) for row in avg]
    return [r[0] for r in res], np.array([r[1] for r in res])


def compute_average(window):
    """Media de una ventana de CIRs. window: iterable de arrays complejos (n_rx x n_taps)."""
    if len(window) == 0:
        return None
    arr = np.array(window)[..., cfg.SAMPLE_OFFSET:]  # (N, n_rx, n_taps - offset): descarta las primeras muestras
    if cfg.AVG_MODE == "coherent":
        return np.abs(arr.mean(axis=0))
    return np.abs(arr).mean(axis=0)


def _reset_unlocked():
    st.win_raw.clear()
    st.win_clean.clear()
    st.avg_state["frozen"] = False
    reset_presence()
    for f in st.clutter_filters.values():
        f.reset()


def reset_window():
    with st.live_lock:
        _reset_unlocked()

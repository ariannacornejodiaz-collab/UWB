"""
Simulacion con datos sinteticos usando el codigo real (ClutterFilter + PresenceDetector). No necesita radar.

    python simulacion_visual.py [snr=14] [seed=1] [empty] [thr=2.0] [tau=5]

Escenario (150 s a 20 notif./s): sala vacia 20 s -> una persona entra a 5 m y camina despacio (0.15 m/s) hasta 3 m ->
se queda PARADA 37 s (solo respira y se balancea) -> se aleja despacio (0.17 m/s) hasta 5.5 m -> parada 30 s -> sale.
Con `empty` se simula 10 minutos de sala vacia (mide falsas alarmas).

Modelo: clutter estatico fuerte + deriva/jitter de fase y ganancia comun + jitter por tap proporcional al clutter + ruido
gaussiano. La persona es un torso (respiracion de 3 mm a 0.27 Hz + balanceo de ~1 mm) y unos brazos con movimiento
aleatorio; la fase del eco sigue 4*pi*d/lambda (lambda = 3.75 cm en el canal 9), asi que al caminar la fase gira rapido.
SNR = potencia del eco / potencia de ruido por tap y notificacion a 2 m, con caida 40*log10(d) (ley del radar).
Guarda simulacion_cir.png (misma figura que la sesion real: CIR arriba, distancia abajo).
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import serial  # noqa
except ImportError:                                   # solo para poder importar los modulos sin pyserial
    sys.modules["serial"] = types.SimpleNamespace(Serial=lambda: None, to_bytes=bytes)

import matplotlib
if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")
import numpy as np

import clutter
import config as cfg
import presence_detect as pdm
import track_plot

OFF, MPS, N, DT, SIG = cfg.SAMPLE_OFFSET, cfg.METERS_PER_SAMPLE, 120, cfg.RANGING_INTERVAL_MS / 1000.0, 15.0
LAMBDA = 0.0375
KEY = [(20, 5.0), (33, 3.0), (70, 3.0), (85, 5.5), (115, 5.5)]       # (t, distancia) de la persona; fuera de rango = sala vacia


def true_d(t):
    return float(np.interp(t, [k[0] for k in KEY], [k[1] for k in KEY])) if KEY[0][0] <= t <= KEY[-1][0] else float("nan")


def simulate(snr2m=14.0, seed=1, t_end=150.0, empty=False):
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(1)
    CL = np.zeros((2, N), complex)                    # clutter estatico (acoplamiento directo + paredes + muebles)
    for rx in range(2):
        for t, a in [(3, 2500), (5, 6000), (6, 4000), (18, 3000), (30, 1800), (45, 1500), (70, 1000), (90, 800)]:
            CL[rx, t] += a * np.exp(1j * g.uniform(0, 6.28))
    n = int(t_end / DT)
    times = np.arange(n) * DT
    taps = np.arange(N)
    phi_g, sway, arm = 0.0, np.zeros(2), rng.uniform(0, 6.28, 2)
    psi = rng.uniform(0, 6.28, 2)
    frames, truth = [], []
    for k in range(n):
        t = times[k]
        d = float("nan") if empty else true_d(t)
        truth.append(d)
        phi_g += rng.normal(0, 0.002)                                     # deriva lenta de fase
        sway = 0.97 * sway + rng.normal(0, 0.00025, 2)                    # balanceo ~1 mm, paso bajo
        arm += rng.normal(0, 0.15, 2)                                     # brazos: fase aleatoria
        out = np.zeros((2, N), complex)
        for rx in range(2):
            x = CL[rx] * (1 + 0.003 * (rng.normal(size=N) + 1j * rng.normal(size=N)))     # jitter por tap
            if not np.isnan(d):
                snr = snr2m - 40 * np.log10(d / 2.0)
                amp = np.sqrt(10 ** (snr / 10) * 2 * SIG ** 2)
                disp = 0.003 * np.sin(2 * np.pi * 0.27 * t + psi[rx]) + sway[rx]            # pecho + balanceo (m)
                t0 = OFF + d / MPS
                ph = 4 * np.pi * (d + disp) / LAMBDA
                x = x + amp * np.exp(-0.5 * ((taps - t0) / 1.0) ** 2) * np.exp(1j * ph)
                x = x + 0.4 * amp * np.exp(-0.5 * ((taps - t0 - 1.0) / 1.0) ** 2) * np.exp(1j * arm[rx])
            noise = rng.normal(0, SIG, N) + 1j * rng.normal(0, SIG, N)
            out[rx] = np.round(x * np.exp(1j * (phi_g + rng.normal(0, .25))) * (1 + rng.normal(0, .02)) + noise)
        frames.append(out)
    return times, np.array(truth), np.array(frames)


def run_detector(frames, times):
    """Pasa las notificaciones por el ClutterFilter y el PresenceDetector reales (sin fondo: calentamiento de 32 notif.)."""
    fl = [clutter.ClutterFilter() for _ in range(frames.shape[1])]
    det = pdm.PresenceDetector()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(len(frames)):
            c = np.array([fl[i].process(frames[k][i]) for i in range(len(fl))])
            if all(f.ready for f in fl):
                det.update(c, times[k], k)
    return det


def metrics(h, times, truth, empty=False):
    t, pres, trk = h["t"], h["present"].astype(bool), h["trk_m"]
    td = np.interp(t, times, np.nan_to_num(truth, nan=-1.0))
    td[np.interp(t, times, np.isnan(truth).astype(float)) > 0.5] = np.nan
    print("evaluaciones: %d | tiempo PRESENTE: %.1f %%" % (len(t), 100 * pres.mean()))
    if empty:
        n_ev = int(np.sum(np.diff(np.r_[0, pres.astype(int)]) == 1))
        print("sala vacia: %d falsas detecciones | score pico maximo %.2f (umbral ON %.1f)" % (n_ev, h["score"].max(), cfg.PRESENCE_THR_ON))
        return
    here = ~np.isnan(td)
    first = np.flatnonzero(pres & (t >= KEY[0][0]))
    print("persona entra en t=%.0f s -> PRESENTE a los %.1f s" % (KEY[0][0], t[first[0]] - KEY[0][0]) if len(first) else "NO se detecta la entrada")
    gone = np.flatnonzero((~pres) & (t > KEY[-1][0]))
    print("persona sale en t=%.0f s -> AUSENTE a los %.1f s" % (KEY[-1][0], t[gone[0]] - KEY[-1][0]) if len(gone) else "NO se declara ausente")
    fa = pres & ~here & (t < KEY[0][0])
    print("falsas alarmas con sala vacia al inicio: %.1f s" % (fa.sum() * np.median(np.diff(t))))
    for name, a, b in [("camina 5->3 m", 26, 33), ("PARADA a 3 m", 40, 70), ("se aleja 3->5.5 m", 75, 85), ("PARADA a 5.5 m", 90, 115)]:
        s = (t >= a) & (t < b) & here
        e = np.abs(trk[s] - td[s])
        e = e[np.isfinite(e)]
        cov = 100 * np.mean(pres[s]) if s.any() else 0
        print("%-18s detectada %5.1f %% del tiempo | error de distancia: mediana %.2f m, p90 %.2f m" % (
            name, cov, np.median(e) if len(e) else np.nan, np.percentile(e, 90) if len(e) else np.nan))


if __name__ == "__main__":
    snr, seed, empty = 14.0, 1, False
    for a in sys.argv[1:]:
        if a.startswith("snr="):
            snr = float(a[4:])
        elif a.startswith("seed="):
            seed = int(a[5:])
        elif a == "empty":
            empty = True
        elif a.startswith("thr="):
            cfg.PRESENCE_THR_ON = float(a[4:]); cfg.PRESENCE_THR_OFF = max(1.0, 0.7 * cfg.PRESENCE_THR_ON)
        elif a.startswith("tau="):
            cfg.PRESENCE_TAU_S = float(a[4:])
    times, truth, frames = simulate(snr, seed, 600.0 if empty else 150.0, empty)
    det = run_detector(frames, times)
    h = det.history()
    print("SNR a 2 m = %.1f dB (a 3 m %.1f dB, a 5.5 m %.1f dB) | umbral ON/OFF = %.1f/%.1f | tau = %.1f s" % (
        snr, snr - 40 * np.log10(1.5), snr - 40 * np.log10(2.75), cfg.PRESENCE_THR_ON, cfg.PRESENCE_THR_OFF, cfg.PRESENCE_TAU_S))
    metrics(h, times, truth, empty)
    if not empty:
        png = "simulacion_cir.png"
        track_plot.save_session_figure(h, png, "Simulacion: entra a 5 m, camina a 3 m, PARADA, se aleja a 5.5 m, PARADA, sale (t=115 s)  |  SNR(2 m) = %.0f dB" % snr,
                                       truth=(times, truth), t_ref=times[0])
        print("Guardado:", png)

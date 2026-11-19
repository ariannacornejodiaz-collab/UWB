"""
Reanaliza una grabacion (carpeta cir_*) con el detector actual, SIN radar. Sirve para ajustar umbrales con datos reales.

    python replay.py cir_261103_123456 [thr=2.0] [tau=5] [offset=6] [raw] [bg=background.npz]

Entrada: cir_clean_all.npy (la entrada exacta del detector durante la captura) y los tiempos de cir_dist_aoa.csv.
         Con `raw` (o si faltan/no cuadran) se vuelve a pasar cir_all.npy por el ClutterFilter.
Salida (en esa carpeta): replay_track.png (CIR arriba, distancia abajo; en gris la distancia del firmware del radar) y replay_track.csv
"""
import contextlib
import io
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

import config as cfg
import track_plot
from clutter import ClutterFilter
from csv_export import write_track_csv
from presence_detect import PresenceDetector


def load_times_and_firmware(folder, n):
    path = os.path.join(folder, "cir_dist_aoa.csv")
    if os.path.exists(path):
        d = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2, 3))
        if d.ndim == 2 and len(d) == n:
            return d[:, 0], d[:, 1], d[:, 2]
    t0 = 0.0
    return t0 + np.arange(n) * cfg.RANGING_INTERVAL_MS / 1000.0, None, None


def clean_frames(folder, raw, bg):
    cir = np.load(os.path.join(folder, "cir_all.npy"))                    # (N, n_rx, n_taps) complejo
    clean_path = os.path.join(folder, "cir_clean_all.npy")
    if not raw and os.path.exists(clean_path):
        clean = np.load(clean_path)
        if len(clean) == len(cir):
            return clean, cir.shape[0]
        print("cir_clean_all.npy no cuadra con cir_all.npy (%d vs %d): se recalcula el clutter removal" % (len(clean), len(cir)))
    fl = [ClutterFilter() for _ in range(cir.shape[1])]
    if bg is not None:
        for i, f in enumerate(fl):
            f.set_background(bg["ref"][i], bg["sigma"][i])
    return np.array([[fl[i].process(cir[k][i]) for i in range(cir.shape[1])] for k in range(len(cir))]), len(cir)


def main():
    folder, raw, bg = None, False, None
    for a in sys.argv[1:]:
        if a.startswith("thr="):
            cfg.PRESENCE_THR_ON = float(a[4:])
            cfg.PRESENCE_THR_OFF = max(1.0, 0.75 * cfg.PRESENCE_THR_ON)
        elif a.startswith("tau="):
            cfg.PRESENCE_TAU_S = max(1.0, float(a[4:]))
        elif a.startswith("offset="):
            cfg.SAMPLE_OFFSET = int(a[7:])
        elif a == "raw":
            raw = True
        elif a.startswith("bg="):
            z = np.load(a[3:])
            bg = {"ref": z["ref"], "sigma": z["sigma"]}
        else:
            folder = a
    if not folder or not os.path.isdir(folder):
        sys.exit(__doc__)
    clean, n = clean_frames(folder, raw, bg)
    times, fw_pres, fw_dist = load_times_and_firmware(folder, n)
    det = PresenceDetector()
    start = 0 if (raw and bg is not None) else cfg.WARMUP_FRAMES                # el detector solo ve notificaciones con el filtro listo
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(start, n):
            det.update(clean[k], times[k], k)
    h = det.history()
    if len(h["t"]) == 0:
        sys.exit("Grabacion demasiado corta: el detector necesita unos %.0f s de calentamiento." % (2 * max(cfg.PRESENCE_TAU_S, 1)))
    t_ref = times[0]
    pres = h["present"].astype(int)
    print("%d notificaciones, %d evaluaciones | thr=%.1f/%.1f tau=%.1f s | PRESENTE %.1f %% del tiempo | score pico max %.2f" % (
        n, len(h["t"]), cfg.PRESENCE_THR_ON, cfg.PRESENCE_THR_OFF, cfg.PRESENCE_TAU_S, 100 * pres.mean(), h["score"].max()))
    for i in np.flatnonzero(np.diff(np.r_[0, pres]) != 0):
        print("  t=%7.1f s  %s%s" % (h["t"][i] - t_ref, "PRESENTE" if pres[i] else "AUSENTE",
                                      "  a %.2f m" % h["trk_m"][i] if pres[i] and np.isfinite(h["trk_m"][i]) else ""))
    fw = (times[fw_pres == 1], fw_dist[fw_pres == 1] / 100.0) if fw_pres is not None else None
    png = os.path.join(folder, "replay_track.png")
    track_plot.save_session_figure(h, png, "%s - replay (thr %.1f, tau %.1f s)" % (os.path.basename(os.path.abspath(folder)),
                                                                                    cfg.PRESENCE_THR_ON, cfg.PRESENCE_TAU_S),
                                   firmware=fw, t_ref=t_ref)
    write_track_csv(h, os.path.join(folder, "replay_track.csv"), os.path.basename(os.path.abspath(folder)), cfg.label, cfg.scenario_name)
    print("Guardado:", png)


if __name__ == "__main__":
    main()

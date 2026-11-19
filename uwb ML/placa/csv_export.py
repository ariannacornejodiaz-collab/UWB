"""Exportacion al terminar: dataset CIR + distancia/AoA (.npz, .csv) y CSV listo para ML."""
import os

import numpy as np

import config as cfg
import state as st
from presence import align_presence_to_cir


def save_dist_aoa():
    """Guarda CIR + distancia + AoA alineados: cir_dist_aoa.csv, .npy sueltos y dataset_cir_dist_aoa.npz."""
    ct, present, dist, aoa, gap_ms = align_presence_to_cir()
    n = len(ct)
    with open(os.path.join(st.cir_dir, "cir_dist_aoa.csv"), "w") as f:
        f.write("cir_idx,time_epoch,presence,dist_cm,aoa_deg,gap_ms\n")
        for i in range(n):
            f.write("%d,%.3f,%d,%.0f,%.0f,%.1f\n" % (i, ct[i], present[i], dist[i], aoa[i], gap_ms[i]))
    np.save(os.path.join(st.cir_dir, "cir_dist_cm.npy"), dist)
    np.save(os.path.join(st.cir_dir, "cir_aoa_deg.npy"), aoa)
    np.save(os.path.join(st.cir_dir, "cir_presence.npy"), present)
    lens = set(len(c) for c in st.cir_all)
    m = min(lens)
    np.savez(os.path.join(st.cir_dir, "dataset_cir_dist_aoa.npz"),
             cir=np.array([c[:m] for c in st.cir_all]),      # (N, n_rx, n_taps) complejo
             dist_cm=dist, aoa_deg=aoa, presence=present, time_epoch=ct, gap_ms=gap_ms)
    n_ok = int(np.sum(~np.isnan(dist)))
    print("Saved dist/AoA for %d/%d CIRs to %s" % (n_ok, n, os.path.join(st.cir_dir, "dataset_cir_dist_aoa.npz")))


def save_ml_csv():
    """
    CSV listo para ML: una fila por CIR (notificacion). Columnas:
      meta/etiquetas: session, scenario, idx, time_epoch, seq, presence, valid, dist_cm, aoa_deg, gap_ms
      features:       rx<k>_i_<t>, rx<k>_q_<t>  (I/Q crudo, enteros, taps desde SAMPLE_OFFSET; tap 0 = 0 m)
                      pdoa_<t>                  (fase RX1-RX2 en rad, [-pi, pi], por tap; solo si hay >= 2 RX)
    valid = 1 si la fila tiene distancia y AoA (presencia detectada y medida alineada).
    Salida: ml_dataset.csv (todas las filas) y ml_dataset_valid.csv (solo valid=1).
    """
    ct, present, dist, aoa, gap_ms = align_presence_to_cir()
    n = len(st.cir_all)
    if n == 0 or len(ct) != n:
        return
    shapes = [c.shape for c in st.cir_all]
    shape0 = max(set(shapes), key=shapes.count)                 # forma mas frecuente (n_rx, n_taps)
    keep = np.array([s == shape0 for s in shapes])
    arr = np.array([c for c, k in zip(st.cir_all, keep) if k])[..., cfg.SAMPLE_OFFSET:]   # (N, n_rx, n_taps_eff)
    _, n_rx, n_taps = arr.shape
    idxs = np.arange(n)[keep]
    valid = (~np.isnan(dist)) & (~np.isnan(aoa))

    cols = ["session", "scenario", "idx", "time_epoch", "presence", "valid", "dist_cm", "aoa_deg", "gap_ms"]
    for r in range(n_rx):
        cols += ["rx%d_i_%d" % (r + 1, t) for t in range(n_taps)]
        cols += ["rx%d_q_%d" % (r + 1, t) for t in range(n_taps)]
    if n_rx >= 2:
        cols += ["pdoa_%d" % t for t in range(n_taps)]

    session = os.path.basename(st.cir_dir)
    path_all = os.path.join(st.cir_dir, "ml_dataset.csv")
    path_val = os.path.join(st.cir_dir, "ml_dataset_valid.csv")
    n_val = 0
    with open(path_all, "w") as fa, open(path_val, "w") as fv:
        fa.write(",".join(cols) + "\n")
        fv.write(",".join(cols) + "\n")
        for row, i in enumerate(idxs):
            c = arr[row]
            feats = []
            for r in range(n_rx):
                feats += ["%d" % v for v in c[r].real]
                feats += ["%d" % v for v in c[r].imag]
            if n_rx >= 2:
                feats += ["%.5f" % v for v in np.angle(c[0] * np.conj(c[1]))]
            meta = [session, cfg.scenario_name, str(i), "%.3f" % ct[i], str(int(present[i])), str(int(valid[i])),
                    "%.0f" % dist[i] if valid[i] else "", "%.0f" % aoa[i] if valid[i] else "",
                    "%.1f" % gap_ms[i] if not np.isnan(gap_ms[i]) else ""]
            line = ",".join(meta + feats) + "\n"
            fa.write(line)
            if valid[i]:
                fv.write(line)
                n_val += 1
    print("Saved ML CSV: %d rows (%d valid) x %d columns -> %s" % (len(idxs), n_val, len(cols), path_all))


def write_track_csv(h, path, session="", label="", scenario=""):
    """Una fila por evaluacion del detector: presencia, score y distancia (cruda y suavizada)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("session,label,scenario,t_epoch,cir_idx,present,score,dist_raw_m,dist_m\n")
        for i in range(len(h["t"])):
            f.write("%s,%s,%s,%.3f,%d,%d,%.3f,%.3f,%.3f\n" % (session, label, scenario, h["t"][i], h["idx"][i], h["present"][i],
                                                              h["score"][i], h["raw_m"][i], h["trk_m"][i]))


def save_presence_log():
    """presence_track.csv (una fila por evaluacion: presencia, score y distancia) + presence_track.png (CIR arriba, distancia abajo)."""
    import numpy as np
    import track_plot
    from presence_detect import det
    h = det.history()
    n = len(h["t"])
    if n == 0:
        return
    session = os.path.basename(st.cir_dir)
    path = os.path.join(st.cir_dir, "presence_track.csv")
    write_track_csv(h, path, session, cfg.label, cfg.scenario_name)
    pres = h["present"].astype(bool)
    d = h["trk_m"][pres & np.isfinite(h["trk_m"])]
    print("Saved %d presence evaluations to %s  (presente en %.1f %% del tiempo%s)" % (
        n, path, 100.0 * pres.mean(), ", distancia mediana %.2f m" % np.median(d) if len(d) else ""))
    with st.presence_lock:
        fw = np.array(st.presence_log, dtype=float).reshape(-1, 4)
    fw = (fw[fw[:, 1] == 1, 0], fw[fw[:, 1] == 1, 2] / 100.0) if len(fw) else None
    png = os.path.join(st.cir_dir, "presence_track.png")
    t_ref = st.cir_times[0] if st.cir_times else None
    if track_plot.save_session_figure(h, png, "%s - CIR sin clutter (arriba) y distancia estimada (abajo)" % session, firmware=fw, t_ref=t_ref):
        print("Saved presence figure to %s" % png)

"""Guardado de las medias fijas: .npy + .csv + _echo.json + .png (carpeta cir_*/scenarios/) y comparativa final."""
import os

import numpy as np

import config as cfg
import state as st
from clutter import detect_all


def plot_profiles(profiles, path, title, rx=None, marks=None):
    """profiles: [(label, avg (n_rx x n_taps))]. marks: {rx: [ecos]} (solo para un perfil)."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n_rx = max(p[1].shape[0] for p in profiles)
    rx_list = range(n_rx) if rx is None else [rx]
    fig, axes = plt.subplots(len(rx_list), 1, figsize=(10, 3.6 * len(rx_list)), squeeze=False)
    for ax, r in zip(axes[:, 0], rx_list):
        for label, avg in profiles:
            if r < avg.shape[0]:
                ax.plot(avg[r], label=label)
        if marks and r in marks:
            for e in marks[r]:
                ax.plot(e["tap"], e["amp"], "kv", ms=8)
                ax.annotate("tap %.2f | %.2f m\n%.1f dB" % (e["tap"], e["dist_m"], e["snr_db"]), (e["tap"], e["amp"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.set_xlabel("Sample (tap index, offset %d)" % cfg.SAMPLE_OFFSET)
        ax.set_ylabel("mean |CIR| (clutter removed)")
        ax.set_title("%s - RX%d" % (title, r + 1))
        ax.grid(alpha=.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_snapshot(avg, n_in_window, echoes=None):
    """Guarda la media fija (npy + csv + png + json de ecos) en cir_dir/scenarios/."""
    import json
    st.snap_counter += 1
    label = "%s_%02d" % (cfg.scenario_name, st.snap_counter)
    out_dir = os.path.join(st.cir_dir, "scenarios")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, label + ".npy"), avg)
    np.savetxt(os.path.join(out_dir, label + ".csv"), avg.T, delimiter=",",
               header=",".join("RX%d" % (i + 1) for i in range(avg.shape[0])), comments="")
    if echoes is None:
        echoes = detect_all(avg)[0]
    with open(os.path.join(out_dir, label + "_echo.json"), "w") as f:
        json.dump({"RX%d" % (i + 1): e for i, e in enumerate(echoes)}, f, indent=1)
    st.snapshots.append((label, avg.copy()))
    plot_profiles([(label, avg)], os.path.join(out_dir, label + ".png"),
                  "%s (media %d/%d notif., %s)" % (label, n_in_window, cfg.AVG_WINDOW, cfg.AVG_MODE),
                  marks={i: e for i, e in enumerate(echoes)})
    return label


def save_final_averages():
    """Al terminar: comparativa de los snapshots (medias fijas) guardados en esta sesion."""
    if len(st.snapshots) > 1:
        plot_profiles(st.snapshots, os.path.join(st.cir_dir, "scenarios", "comparison.png"), "Comparativa de escenarios")

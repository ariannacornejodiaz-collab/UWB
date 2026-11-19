"""
Grafica en vivo: ARRIBA el CIR SIN CLUTTER (|residuo| de la ultima notificacion tras restar el fondo, un trazo por RX, desde 0 m),
ABAJO la distancia estimada en funcion del tiempo (0 = ahora). Ademas guarda la media fija de las
primeras N notificaciones (scenarios/).
Teclas:  'b' = capturar fondo con la sala VACIA   'n' = nuevo escenario (reinicia filtros y detector)
"""
import numpy as np

import config as cfg
import state as st
import track_plot
from clutter import compute_average, detect_all, reset_window, start_background_capture
from presence_detect import det
from snapshots import save_snapshot
from util import output


def run_live_plot(handler):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    span = cfg.PLOT_SPAN_S
    nlast = int(span / (cfg.RANGING_INTERVAL_MS / 1000.0 * max(1, cfg.PRESENCE_UPDATE_FRAMES)) * 1.6) + 20
    off, mps = cfg.SAMPLE_OFFSET, cfg.METERS_PER_SAMPLE

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.38})
    lines_cir = [a1.plot([], [], lw=1.2, label="RX%d" % (r + 1))[0] for r in range(2)]
    est_line = a1.axvline(np.nan, color="tab:red", ls="--", lw=1.5, label="distancia estimada")
    xmax = min(cfg.N_TAPS_EFF * mps, 1.25 * cfg.PRESENCE_MAX_M)
    a1.set_xlim(0, xmax)
    a1.set_xlabel("distancia (m)")
    a1.set_ylabel("|CIR| sin clutter")
    a1.grid(alpha=.3)
    a1.legend(loc="upper right", fontsize=8, ncol=3)
    title = a1.set_title("CIR sin clutter")

    (fw_dots,) = a2.plot([], [], ".", color="tab:gray", ms=4, alpha=.6, label="radar (firmware)")
    (raw_dots,) = a2.plot([], [], ".", color="tab:blue", ms=4, alpha=.5, label="medida (pico del score)")
    (trk_line,) = a2.plot([], [], "-", color="tab:red", lw=2, label="distancia estimada")
    shade = {"c": None}
    track_plot.style_bottom(a2)
    a2.set_xlim(-span, 0)
    a2.set_xlabel("tiempo (s)   [0 = ahora]")
    a2.legend(loc="upper right", fontsize=8, ncol=3)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13)
    pres_txt = fig.text(0.5, 0.012, "PRESENCIA: --", ha="center", va="bottom", fontsize=14, fontweight="bold")

    state = {"count": -1, "ymax": 1.0}

    def update(_):
        if handler.sigint or st.session_status.allow_end.is_set():
            plt.close(fig)
            return
        if st.bg_capture["active"]:
            title.set_text("CAPTURANDO FONDO (sala vacia) %d/%d - no te muevas" % (st.bg_capture["n"], cfg.BG_FRAMES))
            title.set_color("tab:red")
            return
        with st.live_lock:
            if st.live_last is None or st.live_count == state["count"]:
                return
            state["count"] = st.live_count
            cur = np.abs(st.live_last_clean)[:, off:]                 # sin clutter y sin las muestras anteriores a 0 m
            win = list(st.win_clean) if not st.avg_state["frozen"] else None
            warm = st.clutter_filters[0].n if 0 in st.clutter_filters else 0
            h = det.history(nlast)
            present, peak, warm_det = det.present, det.score_peak, det.warmup_frac
            t_now = st.cir_times[-1] if st.cir_times else None

        # la media de las primeras N notificaciones no se muestra: solo se guarda en scenarios/
        if win is not None and len(win) >= cfg.AVG_WINDOW:
            st.avg_state["frozen"] = True
            avg = compute_average(win)
            echoes, _ = detect_all(avg)
            label = save_snapshot(avg, len(win), echoes)
            for r, es in enumerate(echoes):
                for e in es:
                    output("[ECHO] RX%d tap=%.2f (muestra %.2f) dist=%.2f m amp=%.0f SNR=%.1f dB" %
                           (r + 1, e["tap"], e["sample"], e["dist_m"], e["amp"], e["snr_db"]))
            output("[AVG] Media fija de las primeras %d notificaciones guardada: %s" % (cfg.AVG_WINDOW, label))

        for r, ln in enumerate(lines_cir):                      # CIR tal cual
            if r < cur.shape[0]:
                ln.set_data(np.arange(cur.shape[1]) * mps, cur[r])
        state["ymax"] = max(0.97 * state["ymax"], float(cur[:, :int(xmax / mps)].max()), 1.0)
        a1.set_ylim(0, 1.15 * state["ymax"])
        title.set_text("CIR sin clutter   |   fondo: %s" % ("CAPTURADO" if st.bg_data is not None else
                       "NO capturado -> con la sala vacia pulsa 'b'"))
        title.set_color("black" if st.bg_data is not None else "tab:red")

        n_ev = len(h["t"])
        if n_ev == 0 or t_now is None:
            pres_txt.set_text("PRESENCIA: calentando el detector %d %%   (mejor con la sala vacia, o pulsa 'b')" % int(100 * warm_det))
            pres_txt.set_color("gray")
            est_line.set_xdata([np.nan, np.nan])
            return
        last_trk = h["trk_m"][-1]
        if np.isfinite(last_trk):
            est_line.set_xdata([last_trk, last_trk])
        else:
            est_line.set_xdata([np.nan, np.nan])
        if warm_det < 1.0:                                      # tras 'b'/'n' el detector se recalibra
            pres_txt.set_text("PRESENCIA: recalibrando el detector %d %%   (sala vacia)" % int(100 * warm_det))
            pres_txt.set_color("gray")
        else:
            pres_txt.set_text("PRESENCIA: %s%s     score %.1f (umbral ON %.1f / OFF %.1f)" % (
                "SI" if present else "NO", ("   a %.2f m" % last_trk) if (present and np.isfinite(last_trk)) else "",
                peak, cfg.PRESENCE_THR_ON, cfg.PRESENCE_THR_OFF))
            pres_txt.set_color("tab:red" if present else "tab:green")

        x = h["t"] - t_now
        pres = h["present"].astype(bool)
        raw_ok = pres & np.isfinite(h["raw_m"])
        raw_dots.set_data(x[raw_ok], h["raw_m"][raw_ok])
        trk_line.set_data(x, h["trk_m"])
        if shade["c"] is not None:
            shade["c"].remove()
        shade["c"] = a2.fill_between(x, 0, 1, where=pres, transform=a2.get_xaxis_transform(), color="tab:red", alpha=.10)
        with st.presence_lock:
            fw = np.array(st.presence_log[-400:], dtype=float).reshape(-1, 4)
        fw = fw[(fw[:, 1] == 1) & np.isfinite(fw[:, 2]) & (fw[:, 0] >= t_now - span)]
        fw_dots.set_data(fw[:, 0] - t_now, fw[:, 2] / 100.0)

    def on_key(event):
        if event.key == "n" and not st.bg_capture["active"]:    # nuevo escenario: reinicia filtros, ventana de la media y detector
            reset_window()
            state["count"] = -1
            output("[AVG] Nuevo escenario: se vuelven a promediar las primeras %d notificaciones" % cfg.AVG_WINDOW)
        elif event.key == "b":                                  # capturar fondo con la escena vacia
            start_background_capture()
            state["count"] = -1
            output("[BG] Capturando fondo con escena vacia (%d notificaciones)..." % cfg.BG_FRAMES)

    fig.canvas.mpl_connect("key_press_event", on_key)
    ani = FuncAnimation(fig, update, interval=cfg.PLOT_INTERVAL_MS, cache_frame_data=False)
    plt.show()
    return ani

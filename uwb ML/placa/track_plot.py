"""
Figura de seguimiento: arriba el CIR (sin clutter) en funcion del tiempo y la distancia, abajo la distancia estimada.
La usan la grafica en vivo (plot_live.py), el PNG final de cada sesion, replay.py y simulacion_visual.py.
"""
import numpy as np

import config as cfg

CIR_RANGE = (0.0, 20.0)         # dB sobre el ruido (mapa del CIR)
SCORE_RANGE = (0.0, 10.0)       # dB del score de deteccion (0 dB = suelo; umbral ON = 10*log10(PRESENCE_THR_ON))


def grid_map(t, cols, t_lo, t_hi, ncols):
    """Coloca las columnas (N, T) en una malla temporal uniforme (T, ncols). Huecos = NaN. Si dos caen en la misma celda gana la ultima."""
    T = cols.shape[1] if cols.ndim == 2 else 0
    out = np.full((T, ncols), np.nan, dtype=np.float32)
    if len(t) == 0 or t_hi <= t_lo:
        return out
    pos = np.floor((np.asarray(t) - t_lo) / (t_hi - t_lo) * ncols).astype(int)
    ok = (pos >= 0) & (pos < ncols)
    out[:, pos[ok]] = cols[ok].T
    return out


def image_extent(t_lo, t_hi, n_taps):
    mps = cfg.METERS_PER_SAMPLE
    return [t_lo, t_hi, -0.5 * mps, (n_taps - 0.5) * mps]       # el tap i esta centrado en i * mps


def make_cmap(name="viridis"):
    import matplotlib
    cm = matplotlib.colormaps[name].copy()
    cm.set_bad("#222222")
    return cm


def style_bottom(ax):
    ax.set_ylim(0, cfg.PRESENCE_MAX_M)
    ax.set_ylabel("distancia (m)")
    ax.grid(alpha=.3)


def save_session_figure(h, path, title="", truth=None, firmware=None, t_ref=None, max_cols=2500):
    """
    PNG de toda la sesion. h = det.history(). truth = (t, d_m) distancia real (solo simulacion).
    firmware = (t, d_m) distancia que reporta el propio radar (opcional).
    t_ref = instante que se toma como t=0 (por defecto, el primer dato del historial).
    """
    import matplotlib.pyplot as plt
    t = h["t"]
    if len(t) < 5:
        return False
    t0 = t[0] if t_ref is None else t_ref
    x = t - t0
    x_lo = float(x[0])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.08})
    n_taps = h["cir_db"].shape[1]
    ncols = int(min(max_cols, max(10, 0.9 * len(t))))
    img = grid_map(x, h["cir_db"], x_lo, x[-1], ncols)
    im = a1.imshow(img, origin="lower", aspect="auto", cmap=make_cmap(), vmin=CIR_RANGE[0], vmax=CIR_RANGE[1],
                   extent=image_extent(x_lo, x[-1], n_taps), interpolation="nearest")
    a1.plot(x, h["trk_m"], color="white", lw=1.2, label="distancia estimada")
    a1.set_ylim(0, cfg.PRESENCE_MAX_M)
    a1.set_ylabel("distancia (m)")
    a1.set_title(title or "CIR sin clutter (arriba) y distancia estimada (abajo)")
    cax = a1.inset_axes([1.01, 0.0, 0.015, 1.0])
    fig.colorbar(im, cax=cax).set_label("dB sobre el ruido")
    a1.legend(loc="upper right", fontsize=8)

    pres = h["present"].astype(bool)
    a2.fill_between(x, 0, 1, where=pres, transform=a2.get_xaxis_transform(), color="tab:red", alpha=.10, label="PRESENTE")
    if firmware is not None and len(firmware[0]):
        a2.plot(firmware[0] - t0, firmware[1], ".", color="tab:gray", ms=3, alpha=.6, label="radar (firmware)")
    if truth is not None:
        a2.plot(truth[0] - t0, truth[1], "k--", lw=1.3, label="distancia real")
    ok = pres & np.isfinite(h["raw_m"])
    a2.plot(x[ok], h["raw_m"][ok], ".", color="tab:blue", ms=3, alpha=.5, label="medida (pico del score)")
    a2.plot(x, h["trk_m"], "-", color="tab:red", lw=1.8, label="distancia suavizada")
    style_bottom(a2)
    a2.set_xlabel("tiempo (s)")
    a2.set_xlim(x_lo, x[-1])
    a2.legend(loc="upper right", fontsize=8, ncol=5)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.95, bottom=0.08)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True

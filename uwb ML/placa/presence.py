"""Notificaciones de presencia del radar (0x69 0x0A): distancia/AoA y su alineado temporal con cada CIR."""
import os
import time
from datetime import datetime

import numpy as np

import config as cfg
import state as st


def save_presence_raw(uci_hdr, uci_payload):
    st.presence_counter += 1
    with open(os.path.join(st.cir_dir, "presence_raw.csv"), "a") as f:
        f.write("%s,%d,%02X%02X,%s\n" % (datetime.now().isoformat(timespec="milliseconds"), st.presence_counter,
                uci_hdr[0], uci_hdr[1], "".join("{:02x}".format(p) for p in uci_payload)))


def record_presence(uci_payload):
    """Parsea la notificacion de presencia (0x69 0x0A) y guarda distancia (cm) y AoA (grados) con su timestamp."""
    if len(uci_payload) < 7:
        return
    present = (uci_payload[6] == 0x01)
    dist = float("nan")
    aoa = float("nan")
    if present and cfg.detection_mode == 0x03 and len(uci_payload) >= 13:
        dist = float((uci_payload[11] << 8) + uci_payload[10])      # cm, little-endian
        a = uci_payload[12]
        aoa = float(a - 256 if a >= 128 else a)                     # int8, grados
    t = time.time()
    with st.presence_lock:
        st.presence_last.update({"t": t, "present": present, "dist_cm": dist, "aoa_deg": aoa})
        st.presence_log.append((t, float(present), dist, aoa))
    if cfg.is_cir_save and st.cir_dir:
        with open(os.path.join(st.cir_dir, "presence.csv"), "a") as f:
            f.write("%.3f,%s,%d,%d,%.0f,%.0f\n" % (t, datetime.fromtimestamp(t).isoformat(timespec="milliseconds"),
                                                    len(st.presence_log), int(present), dist, aoa))


def align_presence_to_cir():
    """Para cada CIR toma la medida de presencia mas cercana en el tiempo (si esta a menos de MAX_PRESENCE_GAP_S)."""
    n = len(st.cir_times)
    present = np.zeros(n, dtype=np.int8)
    dist = np.full(n, np.nan)
    aoa = np.full(n, np.nan)
    gap_ms = np.full(n, np.nan)
    with st.presence_lock:
        log = np.array(st.presence_log, dtype=float).reshape(-1, 4)
    if n == 0 or len(log) == 0:
        return np.array(st.cir_times), present, dist, aoa, gap_ms
    ct = np.array(st.cir_times)
    pt = log[:, 0]
    idx = np.searchsorted(pt, ct)
    il = np.clip(idx - 1, 0, len(pt) - 1)
    ir = np.clip(idx, 0, len(pt) - 1)
    best = np.where(np.abs(pt[ir] - ct) < np.abs(ct - pt[il]), ir, il)
    gap = np.abs(pt[best] - ct)
    ok = gap <= cfg.MAX_PRESENCE_GAP_S
    present[ok] = log[best[ok], 1].astype(np.int8)
    dist[ok] = log[best[ok], 2]
    aoa[ok] = log[best[ok], 3]
    gap_ms[ok] = gap[ok] * 1000.0
    return ct, present, dist, aoa, gap_ms

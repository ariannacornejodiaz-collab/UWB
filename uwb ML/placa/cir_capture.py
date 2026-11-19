"""Recepcion de notificaciones CIR: fragmentos UCI -> bloques RX -> clutter removal -> ventana de media."""
import os
import time
from datetime import datetime

import numpy as np

import config as cfg
import state as st
from clutter import feed_background, get_filter
from presence_detect import push_frame
from util import output


def extract_cir(byte_array):
    cir_raw = []

    for idx in range(0, len(byte_array), 4):
        cir_sample = byte_array[idx:idx + 4]
        
        real = twos_comp((cir_sample[1] << 8) + cir_sample[0], 16)
        imaginary = twos_comp((cir_sample[3] << 8) + cir_sample[2], 16)
        
        cir_raw.append(complex(real, imaginary))
    
    return np.abs(cir_raw)


def twos_comp(val, bits):
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val


def is_cir_candidate(uci_hdr):
    mt = (uci_hdr[0] >> 5) & 0x07
    gid = uci_hdr[0] & 0x0F
    oid = uci_hdr[1] & 0x3F
    if (mt != 3):
        return False
    if (gid in (0, 1, 2)):
        return False
    if (uci_hdr[0] == 0x69 and uci_hdr[1] == 0x0A):
        return False
    return True


def parse_cir_blocks(data):
    blocks = []
    n_blocks = (len(data) - cfg.PREAMBLE_BYTES) // cfg.BLOCK_BYTES
    for b in range(n_blocks):
        start = cfg.PREAMBLE_BYTES + b * cfg.BLOCK_BYTES
        hdr = data[start:start + cfg.BLOCK_HDR_BYTES]
        raw = data[start + cfg.BLOCK_HDR_BYTES:start + cfg.BLOCK_BYTES]
        iq = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
        cir = iq[:, 0].astype(float) + 1j * iq[:, 1].astype(float)
        seq = int.from_bytes(hdr[0:4], "little")
        rx_tag = hdr[16]
        blocks.append((seq, rx_tag, hdr, cir))
    return blocks


def handle_cir_fragment(uci_hdr, uci_payload):
    
    gid = uci_hdr[0] & 0x0F
    oid = uci_hdr[1] & 0x3F
    pbf = (uci_hdr[0] & 0x10) != 0
    key = (gid, oid)
    
    if (key not in st.cir_buffers):
        st.cir_buffers[key] = bytearray()
    st.cir_buffers[key].extend(uci_payload)
    
    if (pbf):
        return
    
    data = bytes(st.cir_buffers.pop(key))
    st.cir_counter += 1
    name = "ntf_gid%02X_oid%02X_%05d.bin" % (gid, oid, st.cir_counter)
    now = datetime.now().isoformat(timespec="milliseconds")
    
    with open(os.path.join(st.cir_dir, name), "wb") as f:
        f.write(data)
    
    with open(os.path.join(st.cir_dir, "index.csv"), "a") as f:
        f.write("%s,%s,%02X,%02X,%d\n" % (now, name, gid, oid, len(data)))
    
    blocks = parse_cir_blocks(data)
    if (len(blocks) > 0):
        st.cir_all.append(np.array([blk[3] for blk in blocks]))
        st.cir_times.append(time.time())
        cur = np.array([blk[3] for blk in blocks])                       # (n_rx x n_taps) complejo
        with st.live_lock:
            if st.bg_capture["active"]:
                feed_background(cur)                                     # captura de fondo: no entra en la media
                clean = np.zeros_like(cur)
            else:
                filt = [get_filter(i, cur.shape[1]) for i in range(len(cur))]
                ready_before = all(f.ready for f in filt)
                clean = np.array([f.process(cur[i]) for i, f in enumerate(filt)])
                if len(st.win_raw) > 0 and st.win_raw[0].shape != cur.shape:
                    st.win_raw.clear()
                    st.win_clean.clear()
                if ready_before:
                    push_frame(clean, filt, len(st.cir_all) - 1, st.cir_times[-1])
                if ready_before and (not st.avg_state["frozen"]) and len(st.win_clean) < cfg.AVG_WINDOW:
                    st.win_raw.append(cur)
                    st.win_clean.append(clean)
            st.cir_clean_all.append(clean)
            st.live_last = cur
            st.live_last_clean = clean
            st.live_count += 1
        msg = "[CIR] %s (%d bytes)" % (name, len(data))
        for (seq, rx_tag, hdr, cir) in blocks:
            amp = np.abs(cir)
            peak_idx = int(np.argmax(amp))
            power_db = 10 * np.log10(np.sum(amp ** 2) + 1e-9)
            with open(os.path.join(st.cir_dir, "cir_summary.csv"), "a") as f:
                f.write("%s,%s,%d,%d,%d,%.1f,%.2f\n" % (now, name, seq, rx_tag, peak_idx, amp[peak_idx], power_db))
            with open(os.path.join(st.cir_dir, "cir_headers.csv"), "a") as f:
                f.write("%s,%d,%d,%s\n" % (now, seq, rx_tag, "".join("{:02x}".format(x) for x in hdr)))
            msg += "  [seq=%d rx=%d peak_idx=%d peak_amp=%.0f]" % (seq, rx_tag, peak_idx, amp[peak_idx])
        with st.presence_lock:
            pl = dict(st.presence_last)
        if pl["t"] is not None and pl["present"]:
            msg += "  [radar dist=%.0f cm AoA=%.0f deg]" % (pl["dist_cm"], pl["aoa_deg"])
        output(msg)
    else:
        output("[CIR] %s (%d bytes) - too short, saved raw only" % (name, len(data)))

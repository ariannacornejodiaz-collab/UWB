"""
Estado compartido en tiempo de ejecucion (buffers de CIR, ventana de media, filtros de clutter, puerto serie...).

Se usa como `import state as st` y se accede/modifica como `st.NOMBRE`. Al estar todo en un unico modulo,
cualquier modulo puede reasignar una variable (st.cir_dir = ...) y los demas ven el cambio.
"""
from collections import deque
from threading import Condition, Event, Lock

import queue
import serial

import config as cfg

# --- CIR capturado ---
cir_dir = ""
cir_counter = 0
cir_buffers = {}            # (gid, oid) -> bytearray
cir_all = []                # parsed complex CIRs
cir_clean_all = []          # complex CIRs after clutter removal (same shape as cir_all entries)
cir_times = []              # t_epoch de cada CIR (mismo orden/indice que cir_all)

# --- Distancia + AoA reportados por el radar (notificacion de presencia 0x69 0x0A, detection_mode 0x03) ---
presence_counter = 0
presence_lock = Lock()
presence_last = {"t": None, "present": False, "dist_cm": float("nan"), "aoa_deg": float("nan")}
presence_log = []           # [(t_epoch, present, dist_cm, aoa_deg)] una entrada por notificacion de presencia

# --- Plot en vivo ---
live_lock = Lock()
live_last = None            # (n_blocks x n_samples) complex of the latest notification
live_count = 0
live_last_clean = None      # clutter-removed complex CIR of the latest notification

# --- Media de las primeras N notificaciones ---
snap_counter = 0
snapshots = []              # [(label, avg (n_rx x n_taps))] guardados al congelar la media
win_raw = deque(maxlen=cfg.AVG_WINDOW)
win_clean = deque(maxlen=cfg.AVG_WINDOW)
avg_state = {"frozen": False}

# --- Deteccion de presencia: el estado vive en presence_detect.det (PresenceDetector) ---

# --- Clutter ---
clutter_filters = {}        # rx index -> ClutterFilter
bg_data = None              # {"ref": (n_rx x n_taps) complejo, "sigma": (n_rx x n_taps)} fondo de escena vacia
bg_capture = {"active": False, "filters": {}, "n": 0}


# --- Puerto serie / sesion UCI ---
class SessionStates():
    def __init__(self):
        self.allow_config = Event()
        self.allow_start = Event()
        self.allow_stop = Event()
        self.allow_deinit = Event()
        self.allow_end = Event()
    
    def set(self, status):
        if (status == 0x00):
            self.allow_config.set()
            self.allow_start.clear()
            self.allow_stop.clear()
            self.allow_deinit.set()
            self.allow_end.clear()
        
        if (status == 0x01):
            self.allow_config.clear()
            self.allow_start.clear()
            self.allow_stop.clear()
            self.allow_deinit.clear()
            self.allow_end.set()
        
        if (status == 0x02):
            self.allow_config.clear()
            self.allow_start.clear()
            self.allow_stop.set()
            self.allow_deinit.clear()
            self.allow_end.clear()
        
        if (status == 0x03):
            self.allow_config.set()
            self.allow_start.set()
            self.allow_stop.clear()
            self.allow_deinit.set()
            self.allow_end.clear()
        
        if (status == 0xFF):
            self.allow_config.clear()
            self.allow_start.clear()
            self.allow_stop.clear()
            self.allow_deinit.clear()
            self.allow_end.clear()
    
    def set_all(self):
        self.allow_config.set()
        self.allow_start.set()
        self.allow_stop.set()
        self.allow_deinit.set()
        self.allow_end.set()


last_session_started = {0, 0, 0, 0}
serial_port = serial.Serial()
command_queue = queue.Queue(maxsize=100)
session_status = SessionStates()
write_wait = Condition()
go_stop = Event()
stop_write_thread = False
stop_read_thread = False
stop_ipc_thread = False
retry_cmd = False
file_ipc = None
socket = None

"""
Ajustes y constantes del programa.

Se usa siempre como `import config as cfg` y se leen como `cfg.NOMBRE` en el momento de uso, de modo que los
argumentos de linea de comandos (offset=, window=, avg=, scenario=...) pueden modificarlos en tiempo de ejecucion
y todos los modulos ven el cambio.
"""
import numpy as np

### Main variables ###
CHANNEL_NUMBER = 0x09

# Arguments: python main.py [COM12] [notime] [nocir] [noplot] [ipc <prefix_file_name>]
#   Communication Port (e.g. "COM12")
#   Don't put date and time in the log
#   "nocir" to NOT save CIR notifications (also disables the live plot)
#   "noplot" to disable the live CIR plot
#   "ipc" to store all output into file which name is controled by IPC (in this case, timestamp is disabled)
#   "scenario=<name>" label used for the saved averaged CIR snapshots (default "escenario")
#   "window=<N>" number of CIR notifications averaged (default 128)
#   "avg=coherent" average the complex residual (best for weak STATIC echoes; cancels moving ones). Default: abs
#   "bg=<file.npz>" load an empty-scene background saved before (background.npz inside a cir_* folder)
#   "offset=<N>" muestras iniciales del CIR que se descartan (default 6). La muestra N pasa a ser distancia 0 m
#   "label=<x>" etiqueta para el log de presencia (dentro/fuera...)
#   "thr=<x>" umbral de presencia (score = varianza / suelo; default 2.2)   "tau=<s>" ventana de la rama quieta (default 3 s)
# Live plot: averages ONLY the first N notifications (clutter removed), then freezes the plot and saves it.
# Keys: 'b' = capture empty-scene background (best echo sensitivity), 'n' = new scenario (average next first N + recalibrate detector),
#       'v' = top map: CIR without clutter <-> detection score

# Default Port value (COMxx)
com_port = "COM10"

# Log / IPC
is_timestamp = True
is_ipc = False
prefix_ipc = ""

# Presence Detection Config values
detection_mode = 0x03
send_cir_ntf = 0x01      # SEND_RAW_CIR_HOST: 0x01 = send raw CIR to host
enable_irq   = 0x00
sensitivity  = 3.75
distance_min = 50    #cm
distance_max = 800   #cm
hold_delay   = 1600  #ms
min_angle = 0xA6 # -90deg
max_angle = 0x5A # 90deg

# CIR capture settings
is_cir_save = True
CIR_SAMPLES = 128           # must match RADAR_CIR_NUM_SAMPLES (0xA7 en uci_commands.py)
CIR_BYTES_PER_SAMPLE = 4    # int16 real + int16 imag, little-endian

PREAMBLE_BYTES = 10
BLOCK_BYTES = 512
BLOCK_HDR_BYTES = 32
MAX_PRESENCE_GAP_S = 0.5    # un CIR solo se asocia a una medida de presencia si distan menos de este tiempo

# Live plot settings
PLOT_SPAN_S = 30.0          # segundos visibles en la grafica distancia-tiempo (siempre en vivo)
is_plot = True
N_HIST = 200                # waterfall rows
PLOT_INTERVAL_MS = 50
LOG_SCALE = False           # True -> dB in the waterfall

# Averaging settings: media de las primeras AVG_WINDOW notificaciones CIR
AVG_WINDOW = 128            # nº de notificaciones que se promedian
AVG_MODE = "abs"            # "abs" -> mean(|CIR|)   |   "coherent" -> |mean(CIR complejo)|
PLOT_SOURCE = "clutter"     # "clutter" -> media del CIR tras clutter removal   |   "raw" -> CIR sin clutter removal
BLOCK_SAMPLES = (BLOCK_BYTES - BLOCK_HDR_BYTES) // CIR_BYTES_PER_SAMPLE   # taps por bloque RX (120)

# Offset de muestras: el CIR crudo (cir_all.npy, cir_clean_all.npy) se guarda completo, pero el analisis
# (medias, CFAR, graficas, snapshots) empieza en la muestra SAMPLE_OFFSET. Asi tap 0 == muestra SAMPLE_OFFSET
# y la distancia sale corregida (6 muestras ~ 0.9 m  ->  0.15 m por muestra).
SAMPLE_OFFSET = 6
METERS_PER_SAMPLE = 0.9 / 6     # 0.15 m por muestra (distancia radar ida y vuelta)
N_TAPS_EFF = BLOCK_SAMPLES - SAMPLE_OFFSET   # taps utiles tras aplicar el offset
scenario_name = "escenario"

# Clutter removal + deteccion de eco
BG_FRAMES = 128             # notificaciones para capturar el fondo (escena vacia) con la tecla 'b'
WARMUP_FRAMES = 32          # sin fondo: notificaciones para estimar la referencia inicial (no entran en la media)
CLUTTER_ALPHA = 0.999       # adaptacion MUY lenta de la referencia en taps sin eco (deriva termica; ~1000 notif.)
CLUTTER_LEAK = 0.0002       # adaptacion casi nula en taps CON eco (un objetivo quieto no se "absorbe")
GATE_K = 3.0                # tap con eco si |residuo| > GATE_K * sigma                    (ecos fuertes, instantaneo)
SMOOTH_BETA = 0.03          # residuo suavizado (EMA ~33 notif.) para ver ecos PERSISTENTES por debajo del ruido
GATE_KS = 4.0               # tap con eco si |residuo suavizado| > GATE_KS * sigma_suavizada
SIGMA_MIN = 0.5
SIGMA_RATE = 0.005          # adaptacion del ruido sigma por tap (lenta: ~200 notificaciones)
SIGMA_BAND = (0.8, 1.25)    # sigma solo puede variar en esta banda alrededor de su valor calibrado (evita que un eco la infle)
CFAR_TRAIN = 10             # celdas de entrenamiento por lado (OS-CFAR sobre el perfil medio)
CFAR_GUARD = 2              # celdas de guarda por lado
CFAR_K = 4.0                # modo abs: umbral = mediana + CFAR_K * dispersion robusta (MAD)
CFAR_PFA = 1e-8             # modo coherent (ruido Rayleigh): umbral = mediana * sqrt(ln(1/Pfa)/ln2), Pfa por tap
CFAR_RAYLEIGH_FACTOR = float(np.sqrt(np.log(1.0 / CFAR_PFA) / np.log(2.0)))
CFAR_MIN_RATIO = 1.5        # el eco debe superar 1.5x el suelo de ruido local
ECHO_MIN_TAP = 3            # zona ciega del eco CFAR: ignorar taps < ECHO_MIN_TAP (3 taps ~ 0.45 m, resto del acople directo)
MAX_ECHOES = 3

# ======================================================================================================================
# Deteccion de presencia + DISTANCIA (presence_detect.py). Objetivo: persona PARADA o que se mueve despacio.
#
# Una persona parada sigue respirando y balanceandose, y a 8 GHz (lambda ~ 3.8 cm) unos pocos mm bastan para que la fase de su
# eco gire ~1 rad. Lo que delata la presencia es que el eco VARIA en el tiempo, no que supere un umbral en una notificacion
# suelta. Por eso se mide, en cada tap, la VARIANZA TEMPORAL de la senal compleja (ventana exponencial de unos segundos).
# El clutter estatico tiene varianza ~ ruido; la persona, mucha mas. Se normaliza con un suelo local en distancia (CFAR) y
# un suelo aprendido con la sala vacia -> score. Pico del score = distancia de la persona.
# ======================================================================================================================
RANGING_INTERVAL_MS = 50    # intervalo entre notificaciones del radar (0x09 de UWB_SESSION_SET_APP_CONFIG) -> 20 notif./s
PRESENCE_TAU_S = 3.0        # rama QUIETA: constante de tiempo (s) de la ventana exponencial (ventana efectiva ~ 2*tau). Mayor = mas
                            # sensible a gente muy quieta y mas estable, pero mas lenta en declarar presencia / ausencia
PRESENCE_LP_FRAMES = 4      # rama QUIETA: se promedian ~N notificaciones antes de la varianza (el ruido baja; la respiracion pasa)
PRESENCE_TAU_FAST_S = 1.0   # rama MOVIMIENTO (sin paso bajo): ventana corta para quien camina; sigue mejor la posicion
PRESENCE_UPDATE_FRAMES = 2  # se evalua (score, deteccion, distancia) cada N notificaciones (2 -> 10 Hz)
PRESENCE_THR_ON = 2.2       # score minimo para DETECTAR (2.2 = varianza 2.2 veces el suelo, 3.4 dB). Sube si hay falsas alarmas
PRESENCE_THR_OFF = 1.6      # score minimo para MANTENER la presencia (histeresis)
PRESENCE_CONFIRM_S = 0.6    # el score debe superar THR_ON durante este tiempo seguido para declarar PRESENTE
PRESENCE_HOLD_S = 2.0       # tras perder el eco se mantiene PRESENTE este tiempo (evita parpadeo)
PRESENCE_HOLD_FADE_S = 1.0  # retencion CORTA cuando el eco se esta apagando (la persona se ha ido). Mas bajo = declara ausente antes, pero puede perder a gente quieta y lejana
PRESENCE_FADE_RATIO = 0.6   # apagandose: la varianza ABSOLUTA en la posicion cae por debajo de esta fraccion de la de hace DECAY_LAG_S
PRESENCE_FADE_CONFIRM_S = 0.5  # ... y se mantiene asi este tiempo (evita reaccionar a una fluctuacion puntual)
PRESENCE_DECAY_LAG_S = 2.0  # un pico de la rama quieta cuyo exceso de score cae por debajo de DECAY_RATIO del que tenia hace ...
PRESENCE_DECAY_RATIO = 0.8  # ... DECAY_LAG_S segundos es un FANTASMA de alguien que ya se fue: se ignora
PRESENCE_GATE_M = 0.9       # el seguimiento solo acepta picos a menos de esta distancia del anterior ...
PRESENCE_GATE_GROW = 0.3    # ... (+ esta puerta extra por cada segundo sin medida)
PRESENCE_REACQ_FADE_S = 0.4 # reenganche rapido a otro pico fuerte cuando el objetivo actual se esta apagando (evita pasar por NO)
PRESENCE_REACQ_S = 2.0      # si hay un pico fuerte FUERA de la puerta durante este tiempo, se salta a el (reenganche)
PRESENCE_POS_ALPHA = 0.4   # suavizado de la distancia: pos += alpha * (medida - pos) en cada evaluacion
PRESENCE_CFAR_TRAIN = 8     # celdas de entrenamiento por lado para el suelo local en distancia (mediana)
PRESENCE_CFAR_GUARD = 3     # celdas de guarda por lado (el cuerpo ocupa 3-5 taps)
PRESENCE_LEARN_CAL = 100    # evaluaciones (~10 s) iniciales que calibran el suelo por tap SUPONIENDO la sala vacia (o pulsa 'b'/'n' vacia)
PRESENCE_LEARN_MAX = 2.0    # el suelo aprendido no puede superar 2x el suelo local (una persona presente al arrancar no se absorbe)
PRESENCE_NEAR_M = 4.5       # por debajo de esta distancia el clutter es muy fuerte y su jitter tambien: se permite aprender un suelo mucho mayor
PRESENCE_LEARN_MAX_NEAR = 200.0
PRESENCE_LEARN_RATE = 1.0 / 600     # despues de calibrar, aprende despacio (~60 s) y SOLO sin presencia y en taps con score < LEARN_BELOW
PRESENCE_LEARN_BELOW = 1.5
PRESENCE_CIR_SMOOTH = 0.1   # suavizado de la potencia del residuo que se dibuja como CIR (EMA, ~0.5 s)
PRESENCE_MIN_M = 0.5        # zona ciega (el residuo junto al radar no es una persona). Antes 0.8; con 0.5 no hay falsas alarmas en las grabaciones de sala vacia, pero NO esta verificado con alguien a 0,5 m. NXP avisa de que por debajo de 30 cm la distancia no es fiable
PRESENCE_MAX_M = 3.6        # AJUSTADO A TU HABITACION (3,5 m de largo): lo que haya mas lejos son rebotes/multitrayecto (p.ej. el eco a ~4,2 m al entrar), no una persona. Sube a 8.0 en salas grandes
label = ""                  # etiqueta para el log de presencia (p.ej. "dentro"/"fuera"), argumento label=

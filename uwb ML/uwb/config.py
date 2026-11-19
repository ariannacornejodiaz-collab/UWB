"""Constantes compartidas por el lector, el simulador, las características y los modelos.

Los valores medidos salen de data/real/ml_dataset.csv (20 Hz, 114 taps, pico de rx1 en el tap 2).
Los marcados con CONFIRMAR son suposiciones hasta que el grupo de la placa los confirme.
"""

# Formato del CSV de la placa (ml_dataset.csv): 114 taps complejos por antena y trama,
# en las columnas rx1_i_k / rx1_q_k (y rx2_*, que no se usan).
RX = "rx1"            # antena cuyo CIR se usa (CONFIRMAR si conviene rx2 o ambas)
N_TAPS = 114          # taps de cada CIR

# Geometría del CIR
ZERO_TAP = 2          # pico de acoplamiento de rx1 = distancia cero (medido en ml_dataset.csv)
TAP_M = 0.15          # 1 tap = 1 ns = 15 cm de distancia radar (CONFIRMAR)

# Radio
C = 299_792_458.0
FC_HZ = 7.9872e9      # portadora, canal 9 (CONFIRMAR con el grupo de la placa)
WAVELENGTH_M = C / FC_HZ

# Captura
FRAME_RATE_HZ = 20.0  # medido: una trama cada 50 ms. El lector la recalcula con pc_time

# Ventanas de análisis
WINDOW_S = 6.0        # duración de cada ventana
HOP_S = 1.0           # salto entre ventanas consecutivas

# Los primeros centímetros quedan tapados por la deriva del acoplamiento
MIN_RANGE_M = 0.45

# Banda de respiración en reposo
BREATH_BAND_HZ = (0.1, 0.6)

# Escenarios (columna scenario del CSV) y su clase.
# "inside"  = FUERA_ZONA: persona en el pasillo dentro de la zona útil (0,5-3,6 m) -> se abre la puerta
# "outside" = FUERA_NO_ZONA_O_NADIE (nadie / fuera de alcance) y DENTRO (persona dentro de la habitación, casi invisible)
CLASSES = ["inside", "outside"]
POSITIVE = "inside"

SCENARIOS = ["fuera_zona", "fuera_no_zona_o_nadie", "dentro"]
LABEL_MAP = {
    "fuera_zona": "inside",
    "fuera_no_zona_o_nadie": "outside",
    "dentro": "outside",
    "dentro_andando": "outside",
    "dentro_quieta": "outside",
    "dentro_quieto": "outside",
    "vacia": "outside",
    "vacía": "outside",
    "empty": "outside",
    "nadie": "outside",
}


def tap_to_m(tap):
    return (tap - ZERO_TAP) * TAP_M


def m_to_tap(dist_m):
    tap = int(round(ZERO_TAP + dist_m / TAP_M))
    return min(max(tap, 0), N_TAPS - 1)

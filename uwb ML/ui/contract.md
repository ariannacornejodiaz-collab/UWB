# Contrato de datos de la interfaz

La interfaz solo conoce este formato. No importa nada de `uwb\` ni de `placa\`.
Cada fuente (simulación, CSV + modelo, placa en vivo) debe emitir eventos JSON así:

```json
{"t": 1731999999.5, "label": "inside", "p_positive": 0.93,
 "dist_cm": 120, "aoa_deg": -15, "board_present": true,
 "unlock": true, "source": "sim"}
```

| Campo | Tipo | Notas |
|---|---|---|
| `t` | número | segundos epoch |
| `label` | texto | clase libre; la UI no asume cuáles hay |
| `p_positive` | número o null | probabilidad de la clase positiva |
| `dist_cm` | número o null | distancia de la placa; null si no hay medida |
| `aoa_deg` | número o null | ángulo de la placa (positivo = derecha); null si no hay medida |
| `board_present` | bool o null | detector propio de la placa |
| `unlock` | bool | decidido por quien emite, no por la UI |
| `source` | texto | `sim`, `csv`, `live`... |

Los rangos y la clase positiva están en `ui\config.json`.

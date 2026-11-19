# Detección de presencia con radar UWB (SR250): parte de ML

Decide, a partir del CIR de la placa, si hay alguien **en el pasillo, dentro de la zona útil (0,5–3,6 m)**, para abrir la puerta. Las clases internas siguen llamándose `inside`/`outside`, pero ahora significan: `inside` = `fuera_zona` (persona en el pasillo, en zona) y `outside` = `fuera_no_zona_o_nadie` + `dentro` (nadie, fuera de alcance, o persona dentro de la habitación). Escenarios de `scenario=`: `fuera_zona`, `fuera_no_zona_o_nadie`, `dentro`. El texto de abajo describe el esquema anterior (alguien dentro de la sala) y está pendiente de revisar.

(Texto anterior:) Decide, a partir del CIR de la placa, si hay alguien **dentro** de la sala. Compara por separado tres modelos de ML (Random Forest, Gradient Boosting y SVM) y un detector por umbral que sirve de referencia, para poder elegir el que mejor funciona.

Hay dos clases:

- `inside`: hay alguien dentro de la sala.
- `outside`: no hay nadie dentro, porque la sala está vacía o la persona está fuera.

## Puesta en marcha

En PowerShell, desde la carpeta `uwb ML`, activa el entorno que está en la carpeta
superior:

```powershell
& ..\.venv\Scripts\Activate.ps1
```

Si hay que rehacerlo, las dependencias están fijadas a las versiones verificadas
con Python 3.15:

```powershell
python -m venv ..\.venv
& ..\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Con datos simulados (no hace falta la placa):

```powershell
python generate_synthetic.py      # crea data\synthetic (60 grabaciones de una sola sala, ~56 MB)
python train_rf.py                # Random Forest
python train_gb.py                # Gradient Boosting
python train_svm.py               # SVM
python train_umbral.py            # umbral (referencia)
python compare.py                # tabla y gráfica comparando los cuatro
python train_all.py              # lo mismo de arriba en un solo paso
```

Todos los `train_*.py` aceptan las mismas opciones:

| Opción | Qué hace |
|---|---|
| `--data data/real/ml_dataset.csv` | CSV (o carpeta de CSV) en formato `ml_dataset` (por defecto `data/synthetic`) |
| `--tune` | busca hiperparámetros sin mirar las sesiones de test (más lento) |
| `--folds 5` | número máximo de pliegues; debe ser al menos 2 |
| `--no-importance` | se salta el cálculo de importancia de características |
| `--out results` | carpeta de resultados |

Para comparar de forma justa, lanzad los cuatro con las mismas opciones.
`compare.py` rechaza resultados con distintos datos, código, versiones, grupos,
pliegues o uso de `--tune`. Comprueba el contenido de los CSV, no solo sus nombres
o el número de ventanas, y muestra las fechas de entrenamiento.

El proyecto trabaja con **una sola sala**: entrenamiento, validación y prueba
independiente usan la misma sala, y la validación deja fuera sesiones enteras.

La validación necesita al menos dos sesiones. Con `--tune` y con
SVM hacen falta al menos tres sesiones en total, y ambos tipos de clase en cada
entrenamiento interno. La SVM calibra sus probabilidades con sesiones separadas,
incluido el escalado, sin usar la opción obsoleta `SVC(probability=True)`.

Si un entrenamiento falla o sigue en curso, queda `training-incomplete.txt` en
su carpeta de resultados. No se borran los resultados anteriores, pero la
comparación queda bloqueada hasta completar el entrenamiento. Los resultados
antiguos sin huellas verificables también deben regenerarse.

### Error «No hay ventanas» y comprobación de datos

La lectura temporal usa diferencias de fechas en segundos, independientemente
de que pandas almacene fechas en microsegundos o nanosegundos. Una grabación
sintética normal tiene 600 tramas a 20 Hz: genera 25 ventanas de 120 tramas.
No hace falta regenerar los CSV existentes por el fallo temporal de versiones
anteriores.

Se rechazan fechas vacías, repetidas o desordenadas, CIR no finitos, contadores
inválidos y metadatos duplicados. Las pausas mayores
que 1,5 veces el intervalo mediano y las pérdidas de tramas generan un aviso:
la FFT presupone muestreo uniforme; revisa o recorta esas grabaciones.
Una ventana con etiquetas mezcladas se descarta y, si no queda ninguna, el error
indica cuántas candidatas se descartaron.

Pruebas de regresión, sin herramientas adicionales:

```powershell
python -m unittest discover -s tests -v
```

## Qué guarda cada modelo

En `results/<modelo>/`:

| Archivo | Contenido |
|---|---|
| `metrics.json` | exactitud, exactitud equilibrada, F1, AUC, falsas alarmas, «inside» no detectado, acierto por escenario, hiperparámetros |
| `predictions.csv` | la predicción de validación de cada ventana, para buscar dónde falla |
| `confusion.png` | matriz de confusión |
| `importance.csv` / `.png` | cuánto cae el acierto al desordenar cada característica |
| `model.joblib` | el modelo entrenado con todos los datos, listo para tiempo real |

`compare.py` añade `results/comparison.csv` y `results/comparison.png`.

### Cómo leer los resultados

- **Exactitud equilibrada**: la media del acierto en cada clase. Es la métrica principal, porque no se infla si hay más ventanas de una clase.
- **Falsas alarmas**: veces que dice `inside` sin nadie dentro. **Inside no detectado**: veces que no ve a quien está dentro.
- **Acierto por escenario**: dónde falla. Lo esperable es que el peor sea `dentro/still` (persona quieta). Con una sola sala no se mide la generalización a otras salas.
- Si ningún modelo de ML supera con claridad al umbral, quedaos con el umbral.

## Qué hay en cada archivo

| Archivo | Qué hace |
|---|---|
| `uwb/config.py` | Constantes: formato del CSV, tap de distancia cero, cadencia, ventanas, clases y etiquetas |
| `uwb/dataset.py` | Lee `ml_dataset.csv` (varias sesiones por archivo) y construye la tabla de características |
| `uwb/preprocess.py` | Resta de fondo con compensación de deriva, y troceado en ventanas de 6 s |
| `uwb/features.py` | 37 características por ventana: potencia sobre el ruido y distancias absolutas a la placa (no usa la pared del fondo) |
| `uwb/models/random_forest.py` | Random Forest: hiperparámetros por defecto y rejilla de `--tune` |
| `uwb/models/gradient_boosting.py` | Gradient Boosting (`HistGradientBoostingClassifier`) |
| `uwb/models/svm.py` | SVM con kernel RBF, detrás de un `StandardScaler` |
| `uwb/models/threshold.py` | Detector por umbral (referencia DSP, no es ML) |
| `uwb/training.py` | Validación por grupos, métricas, gráficas y guardado; común a los `train_*.py` |
| `uwb/plots.py` | Gráficas |
| `uwb/realtime.py` | `PresenceDetector`: recibe tramas de una en una y devuelve la clase |
| `uwb/simulator.py` | Genera un `ml_dataset.csv` sintético con las mismas 579 columnas que la placa (rx2, pdoa y presence inventados) |
| `inspect_csv.py` | Resumen de una grabación: duración, cadencia, tramas perdidas, etiquetas, pico |
| `plot_session.py` | Mapa distancia-tiempo de una grabación |
| `demo_realtime.py` | Pasa una grabación trama a trama por `PresenceDetector` |

Para cambiar los hiperparámetros de un modelo, se edita solo su archivo en `uwb/models/`.

## Formato de los datos

### `ml_dataset.csv`

Es el formato real de la placa (ejemplo: `data/real/ml_dataset.csv`). Una fila por trama y varias sesiones por archivo:

```
session,scenario,idx,time_epoch,presence,valid,dist_cm,aoa_deg,gap_ms,rx1_i_0…rx1_i_113,rx1_q_0…,rx2_i_0…,rx2_q_0…,pdoa_0…
```

**Columnas que se usan:** `session`, `scenario`, `idx`, `time_epoch`, `rx1_i_*` y `rx1_q_*` (114 taps complejos), y `dist_cm` y `aoa_deg` (opcionales).
**Se ignoran:** `presence`, `valid`, `gap_ms`, `rx2_*` y `pdoa_*`.

- `dist_cm` y `aoa_deg` son estimaciones de la placa; por ventana se calculan media, desviación, rango de la distancia, media y desviación del ángulo y `link_frac` (fracción de tramas con estimación). Si están vacías, valen 0. Se asume que son correctas; el simulador las inventa.
- La clase sale de `scenario`, rellenado a mano por quien graba. `presence` es la salida del detector de la placa, no verdad de terreno (puede ser 0 con una persona quieta), así que no se usa para etiquetar.
- Escenarios admitidos (`LABEL_MAP` en `config.py`): `dentro_andando` y `dentro_quieta` → `inside`; `vacia` y `fuera` → `outside`. Un escenario desconocido (p. ej. el `escenario` del archivo actual) detiene el análisis.
- La cadencia (~20 tramas/s) se calcula con `time_epoch`; los saltos de `idx` se cuentan como tramas perdidas.
- El pico de acoplamiento de rx1 está en el tap 2 (`ZERO_TAP`) y se asumen 15 cm por tap: CONFIRMAR con el grupo de la placa.
- Los nombres de `session` deben ser únicos; la validación cruzada y el holdout separan por sesión, no por fila.
- `data/real/ml_dataset.csv` tiene una sola sesión y `scenario="escenario"`: aún no sirve para entrenar.

## Estado de los datos reales

Para auditar los datos reales:

```powershell
python analyze_dataset.py --data data\real\ml_dataset.csv --out results_real\analysis
```

Guarda `dataset_report.json` con clases, cadencia, irregularidades, pendientes y bloqueos; `signal_windows.csv` contiene medidas de señal residual
por ventana, no las 37 características usadas para entrenar. El código de salida
es 2 si faltan requisitos de entrenamiento, aunque el informe se haya generado.
Las ventanas se cuentan con la cadencia media; las irregularidades no se corrigen
ni se recortan automáticamente. Los 13 s iniciales se usan solo como diagnóstico
exploratorio de arranque, no como un intervalo de descarte recomendado.

Cuando se completen los datos, usa una carpeta de salida separada para no
sobrescribir los modelos sintéticos:

```powershell
python train_all.py --data data\real\ml_dataset.csv --out results_real
```

- `python analyze_dataset.py` y `inspect_csv.py` leen el archivo real; faltan escenarios etiquetados y más sesiones.

## Qué falta cuando lleguen más grabaciones

1. Grabaciones de **sala vacía** y de **persona fuera** (junto a la puerta, abierta y cerrada), además de dentro andando y dentro quieta. Una sola clase y actividad por grabación, varias sesiones, siempre en la misma sala.
2. Rellenar la columna `scenario` de cada sesión con una de las 4 etiquetas.
3. Pasar `inspect_csv.py` y `plot_session.py` a cada grabación nueva antes de entrenar.
4. Confirmar con el grupo de la placa lo marcado con CONFIRMAR en `config.py`: el tap de acoplamiento, los 15 cm por tap y la portadora.
5. `.venv/bin/python train_all.py --data data/real/ml_dataset.csv` y después con `--tune`.
6. Pasar el `model.joblib` elegido al grupo de la interfaz:

```python
from uwb.realtime import PresenceDetector

detector = PresenceDetector("results/gb/model.joblib", frame_rate=20)
result = detector.push(trama)   # trama: 114 taps complejos (rx1)
if result is not None:          # una decisión por segundo, tras los primeros 6 s
    clase, p_inside = result
```

## Qué significan (y qué no) los resultados con datos simulados

### Entrenamiento y prueba independiente

Solo se mantiene un conjunto para entrenar y otro reservado para evaluar:

| Carpeta | Uso |
|---|---|
| `data\synthetic` | Entrenamiento y validación cruzada por sesiones (una sola sala) |
| `results` | Modelos entrenados con `data\synthetic` y métricas de validación cruzada |
| `data\synthetic_holdout` | Prueba independiente: sesiones nuevas de la misma sala, semilla 2026, sin entrenar con ellas |
| `results_holdout` | Métricas, matrices de confusión y predicciones sobre esas sesiones nuevas |

Las carpetas redundantes `data\synthetic_test` y `results_synthetic_test` se han
eliminado. El holdout usa el mismo simulador y la misma sala, pero distinta semilla
y nombres de sesión. Su procedencia está en `data\synthetic_holdout\provenance.json`.
La geometría de la sala es simulada, no medida.

Con los datos ya generados:

```powershell
python train_all.py --data data\synthetic --out results
python evaluate_holdout.py --models results --data data\synthetic_holdout --out results_holdout
```

`evaluate_holdout.py` carga los modelos guardados, sin ejecutar `fit` ni ajustarlos.
Rechaza el propio dataset de entrenamiento, sesiones compartidas,
grabaciones idénticas aunque se hayan renombrado y modelos incompatibles con el
código o las versiones actuales. Los controles no prueban la independencia de
datos manipulados; aquí se garantiza generando sesiones nuevas con otra semilla.
`results_holdout\<detector>\predictions.csv` incluye etiqueta real, predicción,
probabilidad y `correct` para localizar errores.
`results_holdout\comparison.csv` y `.png` comparan el rendimiento independiente.
No se debe pasar esta carpeta a `compare.py`, destinado a validación cruzada.

Para regenerar exactamente el conjunto reservado:

```powershell
python generate_synthetic.py --out data\synthetic_holdout --reps 4 --duration 30 --seed 2026 --name-prefix holdout_
```

No uses `train_all.py --data data\synthetic_holdout` ni elijas hiperparámetros
mirando sus resultados si quieres conservarlo como evaluación independiente.
Tras cambios guiados por estos resultados, reserva otro conjunto nuevo para la
evaluación final. El rendimiento en el mismo simulador no equivale al real.

Para pasar a datos reales se conserva el formato de CSV y metadatos, se completan
las dos clases, y se vuelve a entrenar con
`--data data\real --out results_real`. Cambiar los archivos de entrada no actualiza
un modelo ya guardado. Las métricas sintéticas no predicen su rendimiento real.

- El simulador copia el formato, la cadencia, el acoplamiento y el ruido de las grabaciones reales, pero el eco de la persona y la atenuación de la pared son inventados. **Los porcentajes no dicen nada del rendimiento real**; solo demuestran que el código funciona.
- En el simulador, quien está fuera siempre queda más lejos que la pared del fondo. En una sala real puede estar fuera y más cerca (por un lateral o detrás de la placa); ese caso no está simulado.

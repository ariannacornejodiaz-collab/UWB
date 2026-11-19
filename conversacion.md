# Diagnóstico del proyecto UWB: error al interpretar las fechas

El problema no es que tus datos estén vacíos ni que duren menos de seis segundos: hay un error al interpretar las fechas con la versión de pandas instalada. He revisado los 24 archivos Python del proyecto y comprobado el flujo con tus datos. No he modificado el código ni sobrescrito los resultados.

## 1. Qué provoca exactamente el error

El origen está en `uwb\dataset.py`, líneas 52–54:

```python
times = pd.to_datetime(table["pc_time"]).astype("int64").to_numpy() / 1e9
```

Esta instrucción supone que los enteros de las fechas están expresados en nanosegundos. Sin embargo, con tu instalación de pandas 3.0.6, esas fechas se convierten a `datetime64[us]` (microsegundos).

Al dividir entre mil millones en lugar de interpretar correctamente la unidad, la duración calculada queda mil veces más pequeña y la cadencia mil veces más grande:

| Concepto | Valor correcto | Valor que calcula el código |
| --- | --- | --- |
| Tramas por grabación sintética | 600 | 600 |
| Tiempo entre primera y última trama | 29,95 s | Aproximadamente 0,02995 s |
| Cadencia | 20 tramas/s | Aproximadamente 20.000 tramas/s |
| Tramas necesarias para una ventana de 6 s | 120 | 120.000 |

Como 600 tramas no alcanzan para una ventana de 120.000, `sliding_windows()` no produce ninguna. El programa termina mostrando un mensaje genérico que, en este caso, resulta engañoso.

No necesitas regenerar los CSV ni reducir la ventana. La corrección debe hacerse en la lectura del tiempo, evitando depender de la unidad interna de pandas. Por ejemplo, sustituyendo el cálculo de fechas y duración por:

```python
times = pd.to_datetime(table["pc_time"])
duration = (times.iloc[-1] - times.iloc[0]).total_seconds() if len(times) > 1 else 0.0
frame_rate = (len(times) - 1) / duration if duration > 0 else FRAME_RATE_HZ
```

Esta corrección afecta a todos los entrenamientos porque comparten el mismo lector. También afecta a los tiempos de `inspect_csv.py`, las ventanas de `plot_session.py` y la reproducción de `demo_realtime.py`.

## 2. Cómo está organizado el código

La estructura general es razonable: los modelos no duplican el procesamiento.

| Parte | Archivos | Función |
| --- | --- | --- |
| Configuración | `uwb\config.py` | Define taps, distancias, etiquetas, cadencia y ventanas |
| Generación | `generate_synthetic.py`, `uwb\simulator.py` | Genera grabaciones y metadatos sintéticos |
| Lectura y procesamiento | `uwb\dataset.py`, `uwb\preprocess.py`, `uwb\features.py` | Lee CIR, compensa fondo y deriva, crea ventanas y extrae 20 características |
| Modelos | `uwb\models\*.py` | Define Random Forest, Gradient Boosting, SVM y detector por umbral |
| Entrenamiento | `train_*.py`, `uwb\training.py` | Valida por grupos, calcula métricas y guarda modelos |
| Comparación y gráficas | `compare.py`, `uwb\plots.py` | Presenta los resultados guardados |
| Inspección e inferencia | `inspect_csv.py`, `plot_session.py`, `demo_realtime.py`, `uwb\realtime.py` | Inspecciona señales y realiza predicciones por ventanas |

Un punto positivo importante: la validación separa salas o sesiones completas, en lugar de repartir aleatoriamente ventanas solapadas entre entrenamiento y evaluación. Eso evita una fuente habitual de resultados artificialmente altos.

## 3. Qué ocurre al corregir la cadencia

Con la corrección aplicada solo en memoria, los cuatro detectores completan la validación, sin ajuste de hiperparámetros ni cálculo de importancia.

Los datos sintéticos contienen:

- 60 grabaciones, todas con 600 tramas.
- 6 salas.
- 1.500 ventanas válidas: 600 `inside` y 900 `outside`.
- 20 características por ventana, todas con valores finitos.

| Detector | Exactitud equilibrada |
| --- | --- |
| Gradient Boosting | 98,22 % |
| Random Forest | 98,08 % |
| SVM | 95,11 % |
| Umbral | 91,81 % |

Esto confirma que el bloqueo observado está en la lectura temporal, no en Gradient Boosting. Estos porcentajes corresponden exclusivamente a señales simuladas; no permiten estimar el rendimiento con la placa.

## 4. Otros problemas y limitaciones encontrados

### Entrenamiento con datos reales todavía bloqueado

Las dos grabaciones reales contienen únicamente `dentro`, y ambas tienen `room_depth_m` sin rellenar en `data\real\sesiones.csv`.

Aunque corrijas las fechas, todavía no podrás entrenar correctamente con ellas: faltan la profundidad de sala y grabaciones de la clase `outside`, tanto sala vacía como persona fuera.

### `--tune` no controla todos los casos de validación

En `uwb\training.py:54`, la validación interna se construye así:

```python
inner = GroupKFold(min(3, len(np.unique(groups))))
```

Si solo tienes dos grupos en total, cada partición externa deja un único grupo para entrenar. Al activar `--tune`, se intenta construir `GroupKFold(1)`, que no es válido.

Además, se comprueban las clases de entrenamiento en los pliegues externos, pero no se garantiza que los entrenamientos internos tengan ambas clases. Esto será relevante al incorporar pocas sesiones reales.

### La comparación puede aceptar resultados que no son equivalentes

`compare.py:43` comprueba únicamente `data`, `group` y `n_windows`.

No detecta, por ejemplo, que unos modelos usen distinto número de pliegues o que los CSV hayan cambiado conservando la misma ruta y cantidad de ventanas. Por eso, que no aparezca un aviso no garantiza una comparación idéntica.

### Los resultados anteriores permanecen tras un fallo

Los archivos de `results` que tienes guardados están fechados el 20 de octubre. El intento fallido que mostraste no los actualiza ni los elimina.

Por tanto, ejecutar `compare.py` después del error puede mostrar resultados antiguos. Que existan `model.joblib` y `metrics.json` no significa que el último entrenamiento haya funcionado.

### Compatibilidad y reproducibilidad mejorables

`requirements.txt` no fija versiones. Eso permite cambios de comportamiento entre instalaciones, como el que ha causado este error.

Además, tu versión de scikit-learn ya avisa de que `SVC(probability=True)` está obsoleto y se eliminará en una versión futura. Actualmente funciona, pero requerirá adaptar la calibración de probabilidades.

### Faltan controles de calidad de entrada

El lector obtiene una cadencia media usando solo la primera y última fecha. No comprueba que las fechas sean estrictamente crecientes ni que los intervalos sean regulares.

Si hay pausas o pérdidas, las características espectrales seguirán tratando las tramas como muestras uniformes. También faltan comprobaciones de profundidad positiva y de parámetros válidos para ventanas y cadencia.

### Dos detalles funcionales adicionales

- `train_all.py:17` reconoce `--out carpeta`, pero no interpreta correctamente `--out=carpeta` al lanzar la comparación, aunque los scripts de entrenamiento sí aceptan esa sintaxis.
- `demo_realtime.py` carga el bloque CIR configurado globalmente antes de consultar el bloque guardado en el modelo. Si ambos difieren, podría reproducir el bloque equivocado sin advertirlo.

## 5. Límites del enfoque de detección

La resta de fondo elimina lo que permanece constante dentro de la ventana. Por eso, detectar una persona quieta depende de que su respiración o micromovimientos produzcan señal suficiente; no basta con que exista un eco estático.

Además, el simulador representa a una persona fuera como alguien situado más allá de la pared del fondo. No cubre todas las geometrías reales, como alguien al otro lado de una pared lateral pero a menor distancia del radar. Esa limitación ya está reconocida en el README.

## Conclusión

La prioridad es corregir el cálculo temporal en `uwb\dataset.py`. Después, antes de evaluar con datos reales, hacen falta las profundidades de sala, ejemplos de ambas clases y controles para la validación con pocos grupos. No hay indicios de que debas rehacer el proyecto: el fallo inmediato es concreto y el flujo básico funciona al corregirlo.

# Entrenamiento, guardado de modelos y mejoras propuestas

Sí: el proyecto incluye el entrenamiento y guarda los modelos entrenados. He confirmado que los cuatro archivos existen.

## Entrenamiento y guardado

Desde la carpeta `uwb ML`, con el entorno virtual activado:

```powershell
python train_gb.py       # Entrena Gradient Boosting
python train_all.py      # Entrena los cuatro detectores y los compara
```

El proceso común está en `uwb\training.py` y hace dos cosas distintas:

1. Evalúa el modelo por grupos, dejando salas o sesiones completas fuera del entrenamiento para medir su capacidad de generalizar.
2. Entrena el modelo final con todos los datos disponibles y lo guarda para utilizarlo en predicción.

Los modelos se guardan dentro de tu proyecto:

| Detector | Archivo |
| --- | --- |
| Gradient Boosting | `uwb ML\results\gb\model.joblib` |
| Random Forest | `uwb ML\results\rf\model.joblib` |
| SVM | `uwb ML\results\svm\model.joblib` |
| Umbral de referencia | `uwb ML\results\umbral\model.joblib` |

Cada archivo contiene el estimador entrenado, los nombres de las características, el bloque CIR utilizado y la duración y salto de las ventanas. En esa misma carpeta están `metrics.json`, las predicciones de validación y las gráficas.

Los modelos actuales están entrenados con datos sintéticos, no con las grabaciones reales. Además, ejecutar otra vez el entrenamiento en la misma carpeta sobrescribe el modelo anterior. Puedes conservar ejecuciones separadas así:

```powershell
python train_all.py --out results\experimento_01
```

## Mejoras que priorizaría

| Prioridad | Propuesta | Beneficio |
| --- | --- | --- |
| 1 | Recoger datos reales de sala vacía, persona dentro quieta y andando, y persona fuera; completar profundidades | Es lo más importante: permite saber si el detector funciona realmente, no solo en simulación |
| 2 | Reservar grabaciones de otra fecha o sala como evaluación final independiente | Evita elegir un modelo que solo funciona bien en los datos usados durante el desarrollo |
| 3 | Añadir una herramienta de comprobación del dataset antes de entrenar | Detecta etiquetas, profundidades, pausas, pérdidas y desequilibrios en un único informe |
| 4 | Guardar automáticamente cada entrenamiento en una carpeta fechada, con configuración y métricas | Permite comparar experimentos y recuperar modelos sin sobrescribirlos |
| 5 | Añadir estabilidad temporal a la decisión: histéresis o varias ventanas consecutivas | Reduce cambios continuos entre "dentro" y "fuera"; hay que medir el retraso que introduce |
| 6 | Evaluar distintas ventanas y umbrales de decisión | Permite equilibrar rapidez, falsas alarmas y detección de personas quietas |

## Recomendación

Mi recomendación es priorizar los datos reales y su evaluación antes de añadir modelos más complejos. Después, el versionado automático de entrenamientos sería la mejora de software más útil para trabajar sin perder resultados.

# Comandos para ejecutar y papel de la profundidad de la sala

## 1. Comandos para ejecutar ahora

Desde la carpeta `uwb ML`, activa el entorno:

```powershell
& ..\.venv\Scripts\Activate.ps1
```

### 1.1 Entrenar los cuatro detectores

```powershell
python train_all.py --data data\synthetic --out results
```

Los modelos se guardan en `results\<detector>\model.joblib`.

### 1.2 Evaluarlos con datos nuevos, sin volver a entrenar

```powershell
python evaluate_holdout.py --models results --data data\synthetic_holdout --out results_holdout
```

Muestra aciertos y errores, y guarda la comparación en `results_holdout`.

### 1.3 Ver predicciones segundo a segundo

Persona dentro andando:

```powershell
python demo_realtime.py results\rf\model.joblib data\synthetic_holdout\holdout_room0_dentro_walking_0.csv
```

Persona fuera andando:

```powershell
python demo_realtime.py results\rf\model.joblib data\synthetic_holdout\holdout_room0_fuera_walking_0.csv
```

Sala vacía:

```powershell
python demo_realtime.py results\rf\model.joblib data\synthetic_holdout\holdout_room0_vacia_none_0.csv
```

Puedes sustituir `rf` por `gb`, `svm` o `umbral` para probar otro detector. Los datos ya están generados y los modelos entrenados; puedes empezar directamente por el paso 1.2.

## 2. ¿Es irrelevante la profundidad de la sala?

**Pregunta:** mi profesora me ha dicho que la profundidad de la sala tiene que ser irrelevante, porque el detector tendrá un límite de medida. Además, las medidas son en una sola habitación, y la prueba será en la misma habitación donde se hagan las medidas de entrenamiento. ¿Es correcto?

Sí, puede ser correcto para vuestro planteamiento: la profundidad de la sala no tiene por qué ser una entrada obligatoria si vais a entrenar y utilizar el detector en la misma habitación. Pero hay una distinción importante: el límite de alcance del radar no es lo mismo que el límite de la habitación.

El alcance indica hasta qué distancia el radar obtiene señal utilizable. No garantiza que todo lo que detecte esté dentro: dependiendo de las paredes, la puerta y la colocación, también podría recibir ecos de alguien fuera. Del mismo modo, una persona dentro pero fuera de su zona de cobertura podría no detectarse.

### Qué implica para tu proyecto

Tu código actual sí depende de la profundidad porque fue diseñado así, no porque todos los detectores UWB la necesiten. Utiliza `room_depth_m` para separar taps anteriores y posteriores a la pared y calcular características como la posición de la actividad respecto a ella.

Para una única habitación fija, podemos cambiar ese enfoque:

| Ahora | Enfoque para vuestra prueba |
| --- | --- |
| Separación por profundidad de pared | Zona de análisis fija según el alcance útil del radar |
| Características relativas a la pared | Características de variación, energía y movimiento por distancias dentro de esa zona |


| Evaluación dejando salas fuera | Evaluación dejando sesiones completas fuera, en la misma habitación |

En ese diseño, no sería necesario rellenar `room_depth_m`. El modelo aprendería los patrones de esa habitación con ejemplos reales de persona dentro, sala vacía y persona fuera. No basta con quitar la comprobación de profundidad: habría que modificar las características que la utilizan y volver a entrenar.

### Cómo comprobarlo correctamente

Podéis usar la misma habitación para entrenamiento y prueba; eso encaja con el uso previsto. Lo importante es que la prueba use grabaciones nuevas, preferiblemente de otro momento, sin repartir ventanas solapadas de una misma captura entre entrenamiento y evaluación.

También conviene incluir personas quietas y andando, distintas posiciones dentro y personas fuera junto a la puerta o las paredes relevantes. Así comprobaréis si el radar realmente distingue los casos, en lugar de asumir que su alcance elimina las señales exteriores.

## Conclusión

Tu profesora puede tener razón en que no necesitáis introducir la profundidad. Lo que no podemos asegurar solo por tener un alcance limitado es que el radar distinga automáticamente "dentro" de "fuera". Para vuestro caso, adaptaría el código a una habitación fija y validación por sesiones, sin pretender generalizar a otras salas.
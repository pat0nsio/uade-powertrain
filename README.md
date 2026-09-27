# DPF Health Copilot — Ford Innovation Challenge III (Data-Driven Powertrain Intelligence)

Predicción temprana de eventos de degradación de combustión / filtro de partículas (DPF) a partir de telemetría
de vehículos conectados, con explicación por vehículo y **recomendaciones de manejo que bajan el riesgo**.

> Con la misma tasa de falsas alarmas que la advertencia actual de la ECU, el sistema detecta el **84 %** de los eventos
> antes de que ocurran (IC95 73–93 %), contra el **55 %** de la ECU (IC95 43–68 %). Medido en 198 vehículos que el
> modelo nunca vio, con la política de alerta elegida sin mirar esos vehículos. La validación temporal (sección propia)
> muestra que la ventaja se mantiene hacia el futuro, pero más chica.

## Cómo correrlo

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt   # torch CPU: --index-url https://download.pytorch.org/whl/cpu
# datasets en ./Datasets (estructura original del .zip)
.venv/bin/python -m src.data            # carga + limpieza + regeneraciones reconstruidas (≈20 s) -> data/quality.json
.venv/bin/python -m src.features        # 170 features + etiquetas + test anti-leakage (≈15 s)
.venv/bin/python -m src.models tune     # (opcional) búsqueda de hiperparámetros del GBM, solo folds de train (≈10 min)
.venv/bin/python -m src.models          # ensamble 5 folds + holdout + modelo what-if (≈20 min en CPU)
.venv/bin/python -m src.evaluate        # métricas con IC, lead time vs ECU, SHAP, perfiles
.venv/bin/python -m src.temporal        # validación temporal (despliegue simulado el 2026-01-01)
.venv/bin/streamlit run app.py          # dashboard
```

DuckDB está limitado a 4 GB y 4 hilos: sin límite, las consultas sobre los ~10 M de eventos pueden congelar una
máquina de 16 GB.

## Pipeline

| Paso | Archivo | Qué hace |
|---|---|---|
| Datos | `src/data.py` | DuckDB sobre los CSV (2.5 M viajes, 9.9 M eventos ECU). Limpieza con reporte trazable, reconstrucción de regeneraciones, agregación vehículo-día. |
| Features | `src/features.py` | Ventanas móviles de 7/30/90 días (solo pasado), tendencias, desvío vs historia propia, tasas de vida útil. Etiquetas multi-horizonte con censura. |
| Modelos | `src/models.py` | LightGBM (hiperparámetros ajustados) · Random Survival Forest · GRU multi-tarea con atención · Autoencoder · stacking · GBM monótono para what-if. |
| Evaluación | `src/evaluate.py` | Holdout por vehículo, IC bootstrap por vehículo, umbral fijo y relativo a la flota (elegidos fuera de fold), lead time vs ECU, calibración, SHAP, perfiles. |
| Validación temporal | `src/temporal.py` | Entrena con lo conocido antes de T y evalúa después de T (vehículos nuevos y misma flota); compara umbral fijo vs relativo. |
| Dashboard | `app.py` | Flota · Vehículo (SHAP, supervivencia, atención, what-if, recomendaciones) · Modelo y negocio · Calidad de datos. |

## Hallazgos de datos (calidad y trazabilidad)

1. **Etiquetas v1 inválidas**: en `StaticInformation_FailedVins.csv`, `IdentificationDate == daysUntilSale` en el 76 % de
   los casos. Se usa la versión v2 (`IdentificationDaysSinceProduction`).
2. **Reconstrucción del calendario**: el primer viaje de cada vehículo (en planta) coincide con `D0 + ProductionDay`
   con dispersión p5–p95 de 0.65 días. Eso permite ubicar cada evento en el calendario y alinear la telemetría.
3. **Señal anticipatoria real**: en los 90 días previos al evento los mensajes de DPF "Over Limit/Overloaded" pasan de
   ~0.5 % a ~2.5–2.9 % y las regeneraciones exitosas caen ~40 %; después del service todo vuelve a la normalidad.
4. **Corte de telemetría en el origen, y cómo se recuperó**: desde el 2026-05-26 no llega **ningún** evento
   `Regenerations` en toda la flota, y la bandera ya venía perdiendo eventos desde marzo (1.5–1.9 por 1000 km contra
   ~2.5 reales). Usada tal cual, el riesgo de los sanos subía artificialmente de 4 % a 41 %.
   **Solución**: las regeneraciones se reconstruyen desde la señal de hollín (`Acumulation`), que nunca dejó de llegar.
   Una regeneración es una caída ≥ 20 puntos entre lecturas consecutivas (episodios separados por > 6 h), y la distancia
   entre regeneraciones sale del odómetro. Validado contra la bandera antes del corte: recall 0.91, precisión 0.81,
   correlación vehículo-mes 0.88, y tasa estable antes y después del corte (≈2.5–3 por 1000 km). Así se recupera el
   período completo sin cortes: +37 % de positivos a 30 días. Las caídas de 10–19 puntos se usan como indicio de
   regeneraciones parciales.
5. **Sesgo de cohorte**: los fallados son vehículos más viejos (producidos desde 2023) que los sanos (desde 2025).
   Edad, odómetro y fecha **nunca** son features, y los acumulados se reemplazaron por tasas.
6. Conflictos de etiqueta (27 sanos que figuran como fallados), 19 vehículos con más de un evento (el reloj se reinicia
   tras cada service, con 14 días de exclusión), 40 k viajes duplicados, sentinelas de temperatura (−73/−128 °C).

## Resultados — holdout por vehículo (198 vehículos, 56 eventos)

IC 95 % por bootstrap remuestreando **vehículos** (las filas de un mismo vehículo están correlacionadas).

| Horizonte | AUC holdout (IC95) | AP holdout (base) | AUC fuera de fold (train) | AUC dentro de fallados* |
|---|---|---|---|---|
| 30 días | 0.817 (0.76–0.87) | 0.200 (0.022) | 0.843 | 0.734 |
| 60 días | 0.791 (0.74–0.84) | 0.277 (0.048) | 0.817 | 0.692 |
| 90 días | 0.777 (0.73–0.83) | 0.303 (0.079) | 0.809 | 0.668 |

\* Solo vehículos que fallan: mide si el modelo sabe *cuándo* se acerca el evento, sin poder apoyarse en diferencias
entre cohortes. C-index del modelo de supervivencia: 0.663.

**Anticipación vs falsas alarmas.** Alerta = media móvil de 7 días del riesgo a 90 días ≥ umbral. Episodios con 30 días
de enfriamiento; la anticipación se cuenta hasta 180 días antes del evento. El **punto de operación se elige fuera de
fold**: el mayor umbral cuya tasa de falsas alarmas no supera la de la ECU. Resultado: 10 % de días-sanos en alerta.

Se comparan dos políticas de alerta:
- **Umbral fijo**: probabilidad ≥ un valor fijado en OOF.
- **Umbral relativo a la flota** (política por defecto del dashboard): alerta si el vehículo está en el top X % de riesgo
  de toda la flota en los últimos 30 días. Es causal y no usa etiquetas, así que se calcula en producción; además fija
  el volumen de alertas por diseño, aunque la flota entera se desplace.

| | Detección (IC95) | Anticipación mediana | Falsas alarmas / vehículo-año |
|---|---|---|---|
| Advertencia ECU actual | 55 % (43–68 %) | 100 días | 0.60 |
| Umbral fijo (10 % de días-sanos) | 80 % (70–89 %) | 100 días | 0.61 |
| **Umbral relativo (top 20 % de la flota)** | **84 % (73–93 %)** | 119 días | **0.61** |

La misma comparación fuera de fold (237 eventos de train) da 89 % contra 53 %. La ventaja es en **cuántos** eventos se
detectan; en **días de anticipación** no hay diferencia.

**Qué pesa en el riesgo** (SHAP agrupado): patrón de uso ≈ regeneraciones > vehículo/mercado > hollín > térmico/arranques en frío.

## Validación temporal (despliegue simulado el 2026-01-01)

Solo se entrena con etiquetas que ya se conocían en esa fecha y se evalúa después. Se usa el LightGBM, que es el
componente principal del ensamble.

Los puntos de operación (umbral fijo y relativo) se eligen en el período de calibración [T−90, T), con un modelo
entrenado solo con lo conocido en T−90 y considerando "sano" a todo vehículo sin evento conocido en T.

| Escenario | AUC 30 d (IC95) | AUC 90 d (IC95) | ECU: detección / falsas alarmas | Fijo: det. / FA | Relativo: det. / FA |
|---|---|---|---|---|---|
| Vehículos nuevos (36 eventos) | 0.72 (0.65–0.81) | 0.64 (0.55–0.73) | 53 % / 0.90 | 47 % / 0.54 | 61 % / 0.65 |
| Misma flota (194 eventos) | 0.73 (0.69–0.76) | 0.66 (0.62–0.69) | 43 % / 0.88 | 51 % / 0.39 | 38 % / 0.28 |

**Lectura honesta**:
- Hacia el futuro el desempeño cae (AUC 90 d de ~0.78 a ~0.65), así que el holdout por vehículo es optimista:
  comparte estacionalidad y período con el entrenamiento.
- **A igual tasa de falsas alarmas, fijo y relativo rinden parecido** (con ~0.8 falsas alarmas/año: 61 % vs 67 % en
  vehículos nuevos, 70 % vs 68 % en la misma flota). El umbral relativo no cambia el orden de riesgo, solo cuántas
  alertas salen por día. Las diferencias entre los puntos elegidos vienen de lo ruidoso de la calibración, no del tipo de
  umbral.
- La deriva del umbral fijo es moderada: con objetivo de 10 % de días-sanos en alerta, la tasa real futura es de
  10.6–13.2 %. (Una primera versión de este análisis mostraba el doble; estaba inflada porque la calibración usaba la
  etiqueta final "sano/fallado", que incluye información del futuro.)
- En todos los escenarios el modelo alcanza o supera la detección de la ECU con menos falsas alarmas, pero con vehículos
  nuevos la ventaja está dentro del ruido.

## Modelos

- **LightGBM** por horizonte (30/60/90 d): el componente más fuerte, explicado con SHAP. Se probó una grilla de 8
  configuraciones solo con los folds de train (`models/gbm_params.json`): todas quedan en AUC 0.794–0.803 fuera de fold.
  Se eligió 15 hojas, mínimo 1000 muestras por hoja y L2 = 10. El AUC en entrenamiento sigue en ~0.99 incluso con 7
  hojas: el modelo "reconoce" días de vehículos que ya vio, pero eso no daña la validación, que no cambia con más
  regularización.
- **Random Survival Forest** (landmarking cada 14 días activos): P(evento ≤ t) y días restantes (RUL).
- **GRU multi-tarea con atención** sobre 60 días de telemetría diaria: la atención muestra qué días empujaron el riesgo.
  AUC ≈ 0.72. Es el componente más débil: con ~1 000 vehículos, los árboles sobre features de ingeniería ganan.
- **Autoencoder** entrenado solo con vehículos sanos: su error de reconstrucción es el **Health Index** (0–100).
- **Stacking** logístico sobre las predicciones fuera de fold: calibrado, Brier 0.064 a 90 d.
- **Simulador what-if**: GBM con **restricciones monótonas físicas** (más viajes cortos o regeneraciones interrumpidas
  nunca bajan el riesgo; viajes más largos y rápidos nunca lo suben). Cuesta ~0.01 de AUC y garantiza que cada
  recomendación al cliente sea coherente.

## Limitaciones y próximos pasos

- El desempeño cae hacia el futuro (ver validación temporal): conviene reentrenar periódicamente (por ejemplo, cada
  trimestre) y recalibrar el X % de la política relativa con la capacidad de atención de la red de concesionarios.
- 56 eventos en el holdout y 36 en el temporal: los IC son anchos.
- No hay GPS, presión de neumáticos ni DPF en % en los datos entregados (difieren del anexo de la consigna).
- Los perfiles de conductor (clustering) muestran tasas de falla afectadas por el diseño muestral de las listas de
  fallados/sanos: son descriptivos, no causales.

# DPF Health Copilot — Ford Innovation Challenge III (Data-Driven Powertrain Intelligence)

Predicción temprana de eventos de degradación de combustión / filtro de partículas (DPF) a partir de telemetría
de vehículos conectados, con explicación por vehículo y **recomendaciones de manejo que bajan el riesgo**.

> A igual tasa de falsas alarmas que la advertencia actual de la ECU, el sistema anticipa el **82 %** de los eventos
> (IC95 71–91 %) contra el **50 %** de la ECU (IC95 36–63 %), con una mediana de **124 días** de anticipación.
> Medido en 198 vehículos que el modelo nunca vio.

## Cómo correrlo

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt   # torch CPU: --index-url https://download.pytorch.org/whl/cpu
# datasets en ./Datasets (estructura original del .zip)
.venv/bin/python -m src.data       # carga + limpieza + agregación diaria (≈15 s)  -> data/quality.json
.venv/bin/python -m src.features   # 164 features + etiquetas + test anti-leakage (≈15 s)
.venv/bin/python -m src.models     # ensamble 5 folds + holdout (≈20 min en CPU)
.venv/bin/python -m src.evaluate   # métricas, lead time vs ECU, SHAP, perfiles
.venv/bin/streamlit run app.py     # dashboard
```

## Pipeline

| Paso | Archivo | Qué hace |
|---|---|---|
| Datos | `src/data.py` | DuckDB sobre los CSV (2.5 M viajes, 9.9 M eventos ECU). Limpieza con reporte trazable, agregación vehículo-día. |
| Features | `src/features.py` | Ventanas móviles de 7/30/90 días (solo pasado), tendencias, desvío vs historia propia, tasas de vida útil. Etiquetas multi-horizonte con censura. |
| Modelos | `src/models.py` | LightGBM · Random Survival Forest · GRU multi-tarea con atención · Autoencoder · stacking. |
| Evaluación | `src/evaluate.py` | Holdout por vehículo, lead time vs ECU, IC bootstrap, calibración, SHAP por hipótesis física, perfiles de conductor. |
| Dashboard | `app.py` | Flota · Vehículo (SHAP, supervivencia, atención, what-if, recomendaciones) · Modelo y negocio · Calidad de datos. |

## Hallazgos de datos (calidad y trazabilidad)

1. **Etiquetas v1 inválidas**: en `StaticInformation_FailedVins.csv`, `IdentificationDate == daysUntilSale` en el 76 % de
   los casos. Se usa la versión v2 (`IdentificationDaysSinceProduction`).
2. **Reconstrucción del calendario**: el primer viaje de cada vehículo (en planta) coincide con `D0 + ProductionDay`
   con dispersión p5–p95 de 0.65 días. Eso permite ubicar cada evento en el calendario y alinear la telemetría.
3. **Señal anticipatoria real**: en los 90 días previos al evento los mensajes de DPF "Over Limit/Overloaded" pasan de
   ~0.5 % a ~2.5–2.9 % y las regeneraciones exitosas caen ~40 %; después del service todo vuelve a la normalidad.
4. **Corte de telemetría en el origen**: desde el 2026-05-26 no llega **ningún** evento de regeneración en toda la flota,
   aunque los mensajes ECU siguen llegando. Sin tratarlo, el riesgo de los vehículos sanos subía artificialmente de 4 % a
   41 %. Se trata como faltante (no cero), y entrenamiento y evaluación usan un corte administrativo simétrico en esa fecha.
5. **Sesgo de cohorte**: los fallados son vehículos más viejos (producidos desde 2023) que los sanos (desde 2025).
   Edad, odómetro y fecha **nunca** son features, y los acumulados se reemplazaron por tasas.
6. Conflictos de etiqueta (27 sanos que figuran como fallados), 19 vehículos con más de un evento (el reloj se reinicia
   tras cada service, con 14 días de exclusión), 40 k viajes duplicados, sentinelas de temperatura (−73/−128 °C).

## Resultados (holdout: 198 vehículos, 56 eventos)

| Horizonte | AUC | AP (base) | AUC dentro de fallados* |
|---|---|---|---|
| 30 días | 0.811 | 0.207 (0.022) | 0.747 |
| 60 días | 0.791 | 0.278 (0.046) | 0.709 |
| 90 días | 0.785 | 0.316 (0.071) | 0.692 |

\* Solo vehículos que fallan: mide si el modelo sabe *cuándo* se acerca el evento, sin poder apoyarse en diferencias
entre cohortes. C-index del modelo de supervivencia: 0.676.

**Anticipación vs falsas alarmas** (alerta = media móvil de 7 días del riesgo a 90 días ≥ umbral fijado en OOF;
episodios con 30 días de enfriamiento; anticipación contada hasta 180 días antes del evento):

| Punto de operación | Detección | Anticipación mediana | Falsas alarmas / vehículo-año |
|---|---|---|---|
| Advertencia ECU actual | 50 % | 104 días | 0.52 |
| Modelo al 5 % | 70 % | 111 días | 0.36 |
| **Modelo al 10 % (operación)** | **82 %** | **124 días** | **0.53** |
| Modelo al 20 % | 93 % | 158 días | 0.86 |

**Qué pesa en el riesgo** (SHAP agrupado): patrón de uso > vehículo/mercado ≈ regeneraciones ≈ hollín > térmico/arranques en frío.

**Ablación**: sin variables de vehículo/mercado el AUC baja a 0.78 (30 d) y 0.73 (90 d), pero el AUC dentro de fallados
casi no cambia (0.742 a 30 d). La señal de *cuándo* viene de la telemetría; el país ayuda a decir *qué* vehículo, y puede
reflejar en parte cómo se muestrearon los vehículos.

## Modelos

- **LightGBM** por horizonte (30/60/90 d): el componente más fuerte, explicado con SHAP.
- **Random Survival Forest** (landmarking cada 14 días activos): P(evento ≤ t) y días restantes (RUL).
- **GRU multi-tarea con atención** sobre 60 días de telemetría diaria: la atención muestra qué días empujaron el riesgo.
  Solo es AUC ≈ 0.71; aporta diversidad al ensamble, pero es el componente más débil (con ~1 000 vehículos, los
  árboles sobre features de ingeniería ganan).
- **Autoencoder** entrenado solo con vehículos sanos: su error de reconstrucción es el **Health Index** (0–100).
- **Stacking** logístico sobre las predicciones OOF: calibrado, Brier 0.058 a 90 d.
- **Simulador what-if**: GBM con **restricciones monótonas físicas** (más viajes cortos o regeneraciones interrumpidas
  nunca bajan el riesgo; viajes más largos y rápidos nunca lo suben). Cuesta 0.009 de AUC y garantiza que cada
  recomendación al cliente sea coherente.

## Limitaciones

- 56 eventos en el holdout: la ventaja en detección es significativa, la ventaja en días de anticipación no
  (los IC se superponen).
- No hay GPS, presión de neumáticos ni DPF en % en los datos entregados (difieren del anexo de la consigna).
- Los perfiles de conductor (clustering) muestran tasas de falla afectadas por el diseño muestral de las listas de
  fallados/sanos: son descriptivos, no causales.

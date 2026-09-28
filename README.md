# DPF Health Copilot — Ford Innovation Challenge III (Data-Driven Powertrain Intelligence)

Predicción temprana de eventos de degradación de combustión / filtro de partículas (DPF) a partir de telemetría
de vehículos conectados, con explicación por vehículo y **recomendaciones de manejo que bajan el riesgo**.

> Con la misma tasa de falsas alarmas que la advertencia actual de la ECU, el sistema detecta el **89 %** de los eventos
> antes de que ocurran (IC95 80–96 %), contra el **55 %** de la ECU (IC95 43–68 %). Medido en 198 vehículos que el
> modelo nunca vio, con la política de alerta elegida sin mirar esos vehículos. La validación temporal (sección propia)
> muestra que la ventaja se mantiene hacia el futuro, pero más chica.

## Cómo correrlo

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt   # torch CPU: --index-url https://download.pytorch.org/whl/cpu
# datasets en ./Datasets (estructura original del .zip)
.venv/bin/python -m src.data            # carga + limpieza + regeneraciones reconstruidas (≈20 s) -> data/quality.json
.venv/bin/python -m src.features        # 170 features + etiquetas + test anti-leakage (≈15 s)
.venv/bin/python -m src.models tune     # (opcional) rehace la búsqueda de hiperparámetros del GBM (≈10 min); el
                                        # resultado vigente está versionado en models/gbm_params.json
.venv/bin/python -m src.models          # ensamble 5 folds + holdout + modelo what-if (≈15 min con GPU)
.venv/bin/python -m src.evaluate        # métricas con IC, lead time vs ECU, SHAP, perfiles
.venv/bin/python -m src.temporal        # validación temporal (despliegue simulado el 2026-01-01)
.venv/bin/python -m src.copilot         # chequeos + resumen de receta mínima y ventana de regeneración
.venv/bin/python -m src.edge            # alerta a bordo: reglas y modelos compactos vs ECU -> data/edge.json
.venv/bin/python -m src.edge export     # genera edge/dpf_edge_model.h, compila el C y verifica paridad
.venv/bin/streamlit run app.py          # dashboard

# experimentos con la red neuronal (solo la red; AUC OOF con IC y peso en el stacking -> data/nn_results.jsonl)
.venv/bin/python -m src.models nn <tag> seq=180 tab=1 head=hazard pre=0
.venv/bin/python -m src.temporal nn <tag> seq=180 tab=1 head=hazard pre=0   # la misma red en validación temporal
.venv/bin/python -m src.data v1         # solo para pre=1: agrega los 297 fallados exclusivos de v1 (sin etiqueta)
```

Los recursos por máquina se configuran en `config.local.json` (no versionado; valores por defecto y documentación en
`src/config.py`). Para empezar: `cp config.local.ejemplo.json config.local.json` y ajustar. Cada sección es opcional. DuckDB está limitado por defecto a 4 GB y 4 hilos: sin límite, las consultas
sobre los ~10 M de eventos pueden congelar una máquina de 16 GB (la sección `duckdb` acepta cualquier opción de DuckDB).
La red neuronal usa GPU si hay (`"device": "auto"`); se puede forzar `"cpu"` o `"cuda"` (NVIDIA o AMD con ROCm).

```json
{
  "duckdb": {"memory_limit": "8GB", "threads": 8},
  "torch":  {"device": "auto", "amp": true, "threads": null},
  "gru":    {"seeds": 5, "max_epochs": 40, "patience": 3, "val_frac": 0.15, "batch": 512, "pred_batch": 4096,
             "pre_epochs": 5},
  "env":    {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
}
```

`env` fija variables de entorno antes de inicializar la GPU (el ejemplo hace que ROCm acepte una Radeon RX 6600/6650,
que no está en la lista oficial). Para CPU, instalar PyTorch desde `https://download.pytorch.org/whl/cpu`; para AMD,
desde `https://download.pytorch.org/whl/rocm7.2`.

## Pipeline

| Paso | Archivo | Qué hace |
|---|---|---|
| Datos | `src/data.py` | DuckDB sobre los CSV (2.5 M viajes, 9.9 M eventos ECU). Limpieza con reporte trazable, reconstrucción de regeneraciones, agregación vehículo-día. |
| Features | `src/features.py` | Ventanas móviles de 7/30/90 días (solo pasado), tendencias, desvío vs historia propia, tasas de vida útil. Etiquetas multi-horizonte con censura. |
| Modelos | `src/models.py` | LightGBM (hiperparámetros ajustados) · Random Survival Forest · red híbrida GRU + features con cabeza de riesgo discreto · Autoencoder · stacking · GBM monótono para what-if. |
| Pre-entrenamiento | `src/pretrain.py` | Codificador auto-supervisado (opcional, `pre=1`; evaluado y no adoptado, ver "Red neuronal"). |
| Evaluación | `src/evaluate.py` | Holdout por vehículo, IC bootstrap por vehículo, umbral fijo y relativo a la flota (elegidos fuera de fold), lead time vs ECU, calibración, SHAP, perfiles. |
| Validación temporal | `src/temporal.py` | Entrena con lo conocido antes de T y evalúa después de T (vehículos nuevos y misma flota); compara umbral fijo vs relativo. |
| Copiloto | `src/copilot.py` | Receta mínima (el cambio de hábito más fácil que saca al vehículo de alerta) y ventana de regeneración (cuándo suele hacer un trayecto apto). |
| A bordo | `src/edge.py`, `edge/` | Alerta con memoria fija (EMA) y modelo compacto; exportada a C99 sin memoria dinámica. |
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
| 30 días | 0.829 (0.78–0.88) | 0.201 (0.022) | 0.846 | 0.745 |
| 60 días | 0.803 (0.76–0.86) | 0.285 (0.048) | 0.819 | 0.705 |
| 90 días | 0.785 (0.74–0.83) | 0.314 (0.079) | 0.809 | 0.675 |

\* Solo vehículos que fallan: mide si el modelo sabe *cuándo* se acerca el evento, sin poder apoyarse en diferencias
entre cohortes. C-index del modelo de supervivencia: 0.667.

**Anticipación vs falsas alarmas.** Alerta = media móvil de 7 días del riesgo a 90 días ≥ umbral. Episodios con 30 días
de enfriamiento; la anticipación se cuenta hasta 180 días antes del evento. El **punto de operación se elige fuera de
fold**: el mayor umbral cuya tasa de falsas alarmas no supera la de la ECU. Resultado: 10 % de días-sanos en alerta.

Se comparan dos políticas de alerta:
- **Umbral fijo**: probabilidad ≥ un valor fijado en OOF.
- **Umbral relativo a la flota** (política por defecto del dashboard): alerta si el vehículo está en el top X % de riesgo
  de toda la flota en los últimos 30 días. Es causal y no usa etiquetas, así que se calcula en producción; además fija
  el volumen de alertas por diseño, aunque la flota entera se desplace.

| | Detección (IC95) | Anticipación mediana (IC95 km) | Falsas alarmas / vehículo-año |
|---|---|---|---|
| Advertencia ECU actual | 55 % (43–68 %) | 100 días · 4 313 km (2 214–7 131) | 0.60 |
| Umbral fijo (10 % de días-sanos) | 86 % (75–95 %) | 106 días · 3 368 km (2 241–4 984) | 0.63 |
| **Umbral relativo (top 20 % de la flota)** | **89 % (80–96 %)** | 124 días · 4 492 km (3 322–6 445) | **0.61** |

La anticipación en km es lo recorrido entre la primera alerta y el evento (suma de los km diarios de los viajes), en los
eventos detectados.

La misma comparación fuera de fold (237 eventos de train) da 88 % (fijo) y 90 % (relativo) contra 53 % de la ECU. La ventaja es en **cuántos** eventos se
detectan; en **anticipación** (días o km) no hay diferencia.

**Qué pesa en el riesgo** (SHAP agrupado): regeneraciones ≈ patrón de uso > hollín ≈ vehículo/mercado > térmico/arranques en frío.

## Copiloto del conductor: qué cambiar y cuándo

**Qué viaje sirve para regenerar (medido, no supuesto).** De los 1 857 viajes que arrancan con una regeneración en
curso, la limpieza termina antes de apagar el motor en el 30 % de los de menos de 5 min, 78 % de los de 15–20 min y
**93 % de los de 20 min o más**; la velocidad casi no cambia la tasa. "Trayecto apto" = 20 min o más.

- **Receta mínima**: recorre las combinaciones del simulador (trayectos de ruta por semana, menos viajes cortos, no
  cortar la limpieza) y elige la de menor esfuerzo que lleva el riesgo del GBM monótono por debajo del umbral de la
  flota (top 20 % de los últimos 30 días, la misma regla que la alerta). Si ninguna alcanza, deriva al concesionario.
- **Ventana de regeneración**: con las últimas 12 semanas de viajes del vehículo (solo pasado), en qué día y franja
  suele hacer un trayecto apto. Si hay uno habitual, el consejo es no acortarlo; si solo hay viajes de 10–20 min,
  estirarlos; si no hay ninguno, planificar uno.

En el primer día de alerta de los 118 vehículos del holdout que entran en alerta (52 con evento):
- La receta alcanza en el 85 % de los casos con margen de mejora por manejo, con una reducción mediana del riesgo del
  47 %. En el 23 % el modelo de hábitos no ve margen (el riesgo no viene del manejo reciente): va al concesionario.
- El 58 % no tiene ningún trayecto de 20+ min habitual en la semana; el 39 % sí (la ventana indica cuándo).

Son simulaciones con el modelo, no efectos causales medidos: validar que las recomendaciones evitan eventos requiere
un piloto con intervención.

## Alerta a bordo (ECU / módulo telemático)

A bordo no hay 90 días de historia ni riesgo de la flota: las señales son medias móviles exponenciales (vida media 7 y
30 días; 17 series diarias, proporciones como numerador/denominador) más contadores desde la última regeneración, y el
umbral es fijo. Punto de operación elegido en OOF exigiendo **ni más episodios de falsa alarma ni más días sanos en
alerta** que la ECU (los episodios solos se "ganan" con una alerta casi siempre encendida: avisar con el mensaje
"Full" en vez de "Over Limit" detecta 93 % pero queda encendida en el 58 % de los días sanos).

| | Holdout | Futuro, vehículos nuevos | Futuro, misma flota |
|---|---|---|---|
| ECU actual (detección · FA/veh-año · días sanos en alerta) | 55 % · 0.60 · 5.3 % | 53 % · 0.90 · 5.9 % | 43 % · 0.88 · 7.5 % |
| Modelo compacto, 100 árboles × 7 hojas | **77 %** · 0.53 · 3.5 % | 58 % · 0.74 · 4.3 % | **65 %** · 0.73 · 3.0 % |
| Δ detección vs ECU (IC95) | +21 pts [+5, +36] | +6 [−8, +22] | +23 pts [+13, +31] |

- Tamaño elegido con AUC OOF (0.732, el más chico entre los mejores). Árboles de profundidad 2–4 (la regla legible):
  AUC 0.66–0.69 e inestables hacia el futuro; no superan a la ECU.
- Pasar de ventanas de 90 días a EMA cuesta ~0.05 de AUC (0.78 → 0.73); anticipa menos que el sistema completo.
- **Código C** (`edge/`): C99, sin memoria dinámica ni recursión. Estado: 290 B por vehículo; tablas del modelo:
  ~17 KB en double (la mitad en float32). `dpf_update()` se llama una vez por día. El modelo exportado (un solo modelo,
  umbral con puntajes OOF) da en holdout 80 % de detección con 0.60 FA/veh-año y 4.0 % de días sanos en alerta.
  Paridad con Python: 19 534 días de 40 vehículos, |Δ puntaje| ≤ 9e-16, alertas idénticas.
  `edge/dpf_edge_model.h` se genera con `python -m src.edge export` y no se versiona (modelo entrenado con datos Ford).

## Validación temporal (despliegue simulado el 2026-01-01)

Solo se entrena con etiquetas que ya se conocían en esa fecha y se evalúa después. Las alertas se evalúan con el
LightGBM, que es el componente principal del ensamble; la red neuronal se evalúa aparte (AUC, ver "Red neuronal").

Los puntos de operación (umbral fijo y relativo) se eligen en el período de calibración [T−90, T), con un modelo
entrenado solo con lo conocido en T−90 y considerando "sano" a todo vehículo sin evento conocido en T.

| Escenario | AUC 30 d (IC95) | AUC 90 d (IC95) | ECU: detección / falsas alarmas | Fijo: det. / FA | Relativo: det. / FA |
|---|---|---|---|---|---|
| Vehículos nuevos (36 eventos) | 0.72 (0.63–0.81) | 0.64 (0.55–0.73) | 53 % / 0.90 | 50 % / 0.51 | 64 % / 0.66 |
| Misma flota (194 eventos) | 0.72 (0.69–0.76) | 0.66 (0.63–0.69) | 43 % / 0.88 | 49 % / 0.40 | 38 % / 0.28 |

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
- **Red neuronal híbrida** (GRU con atención sobre 180 días de telemetría diaria + las 170 features del día +
  embeddings de motor, serie y país) con **cabeza de riesgo discreto**: hazard semanal a 26 semanas entrenado con la
  verosimilitud con censura, del que salen P(≤30/60/90 d), coherentes por construcción, y el RUL. Parada temprana por
  AP en un 15 % de vehículos de validación y promedio de 5 semillas (la dispersión entre semillas queda en
  `gru{H}_std`). AUC holdout 0.823 / 0.798 / 0.782: **iguala al LightGBM** (antes 0.72). La atención muestra qué días
  empujaron el riesgo.
- **Autoencoder** entrenado solo con vehículos sanos: su error de reconstrucción es el **Health Index** (0–100).
- **Stacking** logístico sobre las predicciones fuera de fold: calibrado, Brier 0.064 a 90 d.
- **Simulador what-if**: GBM con **restricciones monótonas físicas** (más viajes cortos o regeneraciones interrumpidas
  nunca bajan el riesgo; viajes más largos y rápidos nunca lo suben). Cuesta ~0.01 de AUC y garantiza que cada
  recomendación al cliente sea coherente.

## Red neuronal: experimentos (GPU)

Todas las decisiones con AUC fuera de fold (OOF, IC por vehículo); holdout solo como referencia. Mismas features que la
línea base (se corrieron antes de regenerar los datos, por eso difieren en la tercera cifra de la tabla de Resultados).

| Variante | AUC OOF 30 / 60 / 90 d | AP OOF 90 d | Peso en stacking 90 d |
|---|---|---|---|
| GRU original (60 d, 6 épocas, 1 semilla) | – / – / 0.713 | – | 0.058 |
| + parada temprana y 5 semillas | 0.796 / 0.767 / 0.756 | 0.270 | 0.186 |
| ventana de 180 días | 0.795 / 0.779 / 0.775 | 0.295 | 0.263 |
| + features y embeddings, ventana 60 d | 0.815 / 0.787 / 0.777 | 0.297 | 0.202 |
| + features y embeddings, ventana 180 d | 0.813 / 0.789 / 0.781 | 0.279 | 0.185 |
| **+ cabeza de riesgo discreto, 180 d (elegida)** | **0.827 / 0.793 / 0.778** | 0.289 | 0.194 |
| ídem con ventana de 365 d | 0.821 / 0.792 / 0.780 | 0.290 | 0.197 |
| ídem + pre-entrenamiento auto-supervisado | 0.837 / 0.809 / 0.795 | 0.313 | 0.297 |

- Las variantes con features están todas dentro del ruido entre sí. La de riesgo discreto se eligió por ser la mejor a
  30 d y porque da probabilidades coherentes entre horizontes y RUL. 365 días no aporta y cuesta el doble.
- **Pre-entrenamiento (no adoptado).** Reconstrucción de días enmascarados + predicción de la semana siguiente, con el
  calendario de los vehículos de entrenamiento de cada fold más los 297 fallados que solo están en v1 (nunca holdout, ni
  el fold de validación, ni días posteriores a T). Mejora fuera del ruido en OOF (ΔAUC de a pares +0.010 / +0.016 /
  +0.017, IC95 excluye 0), pero **no en la validación temporal**: ΔAUC +0.013 a +0.015 con vehículos nuevos (IC95
  incluye 0) y +0.015 / +0.003 / −0.002 en la misma flota. No cumple el criterio (mejorar en OOF **y** temporal); queda
  disponible con `pre=1`.
- **Validación temporal de la red** (AUC 30 / 60 / 90 d; entre paréntesis el LightGBM): vehículos nuevos
  0.697 / 0.667 / 0.602 (0.719 / 0.666 / 0.637); misma flota 0.718 / 0.694 / 0.660 (0.724 / 0.693 / 0.658).
  La GRU con parada temprana (60 d, sin features) daba 0.657 / 0.650 / 0.627 y 0.695 / 0.669 / 0.638.
- **Efecto en el ensamble: chico.** El peso de la red en el stacking final sube a 0.41 / 0.32 / 0.23, pero el stack
  mejora poco: ΔAUC de a pares +0.003 / +0.002 / +0.000 en OOF (dentro del ruido) y +0.012 / +0.012 / +0.008 en holdout.
  La red aprendió casi lo mismo que el LightGBM.

## ¿Qué limita el desempeño? (`python -m src.diagnose temporal|learning|retrain|drift`)

**1. La caída temporal es mitad falta de eventos, mitad deriva.** LightGBM, AUC a 90 d en los vehículos del holdout:

| Entrenado con… | Vehículos con evento | Evalúa antes de T | Evalúa desde T |
|---|---|---|---|
| todo el período | 217 | 0.788 | 0.767 |
| todo, 28 % de las filas (mismas filas y positivos que "antes de T") | ~217 | 0.794 | 0.771 |
| todo, pero solo 66 vehículos con evento | 66 | 0.734 | 0.715 |
| solo lo conocido en T (despliegue real) | 66 | 0.770 | **0.653** |

- Recortar **filas** no cuesta nada; recortar **vehículos con evento** a los 66 que había en T cuesta ~0.05.
- El resto (~0.06) es el período: con los mismos 66 vehículos con evento, entrenar solo con datos anteriores a T rinde
  menos después de T (ΔAUC −0.063 [−0.129, +0.007] a 90 d; −0.071 [−0.135, −0.005] a 60 d).
- Por eso la validación temporal es **pesimista para el modelo actual**: en T había 66 vehículos con evento en train;
  hoy hay 217.

**Origen de la deriva.** Las features cambian mucho entre períodos (un clasificador distingue días antes/después de T
con AUC 0.88: vida de aceite, desvíos respecto de la propia historia, temperatura ambiente, país), pero sacar las 5–20
que más cambian **no** mejora el AUC después de T (lo baja a 0.620–0.625). Lo que cambia es la relación con el evento:
la flota se incorporó entre fines de 2024 y fines de 2025, y los eventos por 100 vehículos activos por trimestre pasan
de 1.3 (2025T1) a 5.9 (2025T4) y 8.6 (2026T1); los eventos anteriores a T son de vehículos jóvenes.

**Reentrenar.** Despliegue simulado desde T con reentrenos que usan las etiquetas conocidas a cada fecha (AUC 30 / 60 /
90 d; Δ de a pares contra el modelo estático):

| Escenario | Estático | Trimestral | Mensual |
|---|---|---|---|
| Misma flota | 0.719 / 0.680 / 0.633 | 0.741 / 0.712 / 0.672 (+0.02 a +0.04*) | **0.755 / 0.723 / 0.691** (+0.035 a +0.058*) |
| Vehículos nuevos | 0.722 / 0.664 / 0.653 | 0.699 / 0.677 / 0.653 (≈0) | 0.724 / 0.682 / 0.644 (≈0) |

\* IC95 de la diferencia excluye 0. Reentrenar mensualmente mejora la flota ya monitoreada, pero no a los vehículos
nuevos: la ganancia viene de aprender la historia reciente de esos mismos vehículos, no de corregir la deriva general.
Para vehículos nuevos la palanca es acumular eventos.

**2. Más vehículos sí ayudarían; más días por vehículo no.** Curva de aprendizaje (AUC OOF 90 d; 5 repeticiones para el
LightGBM, 2 para la red):

| Vehículos por fold (fallados) | LightGBM | Red |
|---|---|---|
| 158 (44) | 0.742 ± 0.010 | 0.714 |
| 317 (89) | 0.771 ± 0.007 | 0.742 |
| 477 (134) | 0.788 ± 0.003 | 0.771 |
| 635 (178) | 0.800 | 0.781 |

Cada duplicación de vehículos suma ~+0.03 de AUC y la curva todavía no se aplana (algo menos en el último tramo). La red
necesita más datos que el LightGBM para alcanzarlo. Recortar días manteniendo los vehículos no cuesta nada: la
información está en la cantidad de vehículos y eventos independientes.

## Limitaciones y próximos pasos

- El desempeño cae hacia el futuro, mitad por falta de eventos y mitad por deriva (ver "¿Qué limita el desempeño?"):
  reentrenar **mensualmente** para la flota monitoreada y recalibrar el X % de la política relativa con la capacidad de atención de la red de concesionarios.
- 56 eventos en el holdout y 36 en el temporal: los IC son anchos.
- No hay GPS, presión de neumáticos ni DPF en % en los datos entregados (difieren del anexo de la consigna).
- Los perfiles de conductor (clustering) muestran tasas de falla afectadas por el diseño muestral de las listas de
  fallados/sanos: son descriptivos, no causales.

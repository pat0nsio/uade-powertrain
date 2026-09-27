# Próxima sesión: mejoras de la red neuronal en `pcpat0` (GPU)

Estado al 2026-09-27, commit `6046f5e` + este commit. Objetivo de la sesión: implementar y evaluar las mejoras de la red
neuronal que requieren GPU, **con el mismo protocolo de validación** que el resto del proyecto.

## Punto de partida (para comparar)

| Métrica (holdout por vehículo) | Stack | LightGBM | GRU actual |
|---|---|---|---|
| AUC 30 d | 0.817 | 0.822 | 0.716 |
| AUC 90 d | 0.777 | 0.776 | 0.717 |
| AUC OOF 90 d | 0.809 | 0.803 | 0.713 |

Validación temporal (T = 2026-01-01, LightGBM): AUC 90 d 0.64 (vehículos nuevos) / 0.66 (misma flota).
Peso de la GRU en el stacking: ~0.05 (casi no aporta). Detalle completo en `README.md` y `data/metrics.json`.

---

## 1. Preparar pcpat0

> `data/`, `models/`, `Datasets/` y `duckdb.local.json` **no están en git**: hay que copiarlos o regenerarlos.

1. Verificar la GPU y el disco: `nvidia-smi`, `df -h` (hacen falta ~3 GB libres).
2. Clonar: `git clone https://github.com/pat0nsio/uade-powertrain.git ~/Projects/uade-powertrain`.
3. Copiar los datasets desde `pat0top` por Tailscale. Desde pat0top:
   `scp -r ~/Projects/uade-powertrain/Datasets pcpat0:~/Projects/uade-powertrain/`.
   Son 1.7 GB confidenciales; solo por Tailscale, nunca por servicios externos.
4. Crear `duckdb.local.json` según la RAM de pcpat0, por ejemplo `{"memory_limit": "8GB", "threads": 8}` (usar ~50 % de
   la RAM como máximo).
5. Entorno con PyTorch CUDA:
   ```bash
   uv venv .venv --python 3.12
   uv pip install --python .venv -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 \
       --extra-index-url https://pypi.org/simple
   .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Ajustar `cu124` a la versión de CUDA que muestre `nvidia-smi`. El `requirements.txt` fija `torch==2.14.0`: si no
   hay rueda CUDA para esa versión, usar la más cercana disponible.
6. Hiperparámetros del LightGBM: crear `models/gbm_params.json` con el resultado de la búsqueda ya hecha (o correr
   `python -m src.models tune`, ~10 min):
   ```json
   {"best": {"subsample": 0.8, "subsample_freq": 1, "verbose": -1, "random_state": 0, "n_estimators": 600,
             "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 1000, "colsample_bytree": 0.5,
             "reg_lambda": 10.0, "extra_trees": false}}
   ```
7. Regenerar la línea base y verificar que los números coinciden con la tabla de arriba (pequeñas diferencias numéricas
   son esperables):
   ```bash
   .venv/bin/python -m src.data && .venv/bin/python -m src.features
   .venv/bin/python -W ignore -u -m src.models > data/train.log 2>&1
   .venv/bin/python -W ignore -m src.evaluate > data/eval.log 2>&1
   .venv/bin/python -W ignore -m src.temporal > data/temporal.log 2>&1
   ```

## 2. Tareas, en orden

### 2.0 Soporte de GPU (requisito)
En `src/models.py`:
- `DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Mover `SeqNet` y `AE` a `DEV`.
- En `Seq.fit_scaler`, subir `self.X` a la GPU una sola vez; `Seq.batch` indexa directamente ahí.
- Precisión mixta (`torch.autocast`) en entrenamiento e inferencia; `.cpu().numpy()` al devolver predicciones.
- Tiene que seguir funcionando en CPU sin cambios (pat0top no tiene GPU).

### 2.1 Parada temprana + ensamble de semillas (requisito para medir lo demás)
- En `fit_gru`: separar ~15 % de los **vehículos** del train del fold como validación interna. Parar temprano por AP de
  validación (paciencia ~3 épocas, hasta ~40 épocas) y restaurar los mejores pesos.
- Promediar 5 semillas; guardar la dispersión entre semillas como incertidumbre por fila (`gru90_std`).
- Criterio de éxito: AUC OOF de la GRU > 0.713.

### 2.2 Red híbrida + cabeza de supervivencia discreta (mejora principal)
- **Entradas adicionales** a la cabeza: embeddings de `Engine`, `ModelSeries` y `country`, más las 170 features de
  ingeniería del día de predicción (estandarizadas con estadísticas de train; `NaN` → 0 más máscara).
- **Ventana**: `SEQ` de 60 a 180 días (probar también 365).
- **Cabeza de riesgo discreto** (tipo DeepHit / *discrete-time hazard*): hazard por semana para 26 semanas; la pérdida
  es la verosimilitud con censura (`tte` / `gap_to_end`). De la curva salen P(≤30/60/90 d), coherentes por
  construcción, y el RUL. Mantener las columnas `gru30/60/90` para que el stacking y el dashboard no cambien.
- Criterio de éxito: AUC OOF 90 d de la red ≥ 0.76 y que su peso en el stacking suba. Reportar también holdout y
  temporal.

### 2.3 Pre-entrenamiento auto-supervisado
- Datos sin etiqueta: todo `data/calendar.parquet` **más** los vehículos fallados que solo están en v1
  (`Datasets/TripSummary/TripSummary_Failed_SelectionVins.csv`,
  `Datasets/Dynamic/DynamicInformation_Failed_SelectionVins.csv`; 297 vehículos sin fecha de falla confiable). Hay que
  pasarlos por la misma agregación diaria de `src/data.py`, por ejemplo con una opción para incluir esos archivos
  **solo** para el pre-entrenamiento.
- **Excluir los vehículos del holdout** del pre-entrenamiento, para no contaminar la evaluación.
- Objetivos: reconstrucción de días enmascarados (15–30 %) + predicción de la semana siguiente (hollín, km,
  regeneraciones).
- Nuevo `src/pretrain.py` que guarda el codificador en `models/encoder.pt`; `fit_gru` lo carga y hace el ajuste fino.
- Criterio de éxito: mejora sobre 2.2 fuera del ruido (comparar con IC por vehículo) en OOF **y** en temporal.

### 2.4 Cierre
- Re-correr `src.models`, `src.evaluate` y `src.temporal`; smoke test del dashboard con `streamlit.testing`
  (`AppTest`, las 4 vistas).
- Actualizar `README.md` y `docs/DOCUMENTACION_TECNICA.md` con los resultados, **incluidos los experimentos que no
  mejoraron**.
- Traer a pat0top lo necesario para el dashboard (`models/` y `data/`) si se va a presentar desde ahí.

## 3. Pendientes menores
- El docstring de `Seq` en `src/models.py` dice 90 días; el valor efectivo es `SEQ = 60`.
- Opcional si sobra tiempo: adversarial contra país/cohorte (*gradient reversal*) y modelo jerárquico viajes → días
  (ver la discusión en `docs/DOCUMENTACION_TECNICA.md` §13).

# Sesión GPU en `pcpat0`: hecha (2026-09-27)

El plan anterior (mejoras de la red neuronal con GPU) está completo. Resultados y detalle: `README.md` (secciones
"Resultados" y "Red neuronal") y `docs/DOCUMENTACION_TECNICA.md` §6.4.

## Qué se hizo

| Tarea | Resultado |
|---|---|
| 2.0 Soporte de GPU | Sin acoplar el código: `config.local.json` / `src/config.py` elige `auto`/`cpu`/`cuda`, precisión mixta, hilos y parámetros de la red. pcpat0 tiene una **AMD RX 6650 XT** (no NVIDIA): PyTorch 2.14 + ROCm 7.2 con `HSA_OVERRIDE_GFX_VERSION=10.3.0`. |
| 2.1 Parada temprana + 5 semillas | AUC OOF 90 d 0.713 → 0.756 (criterio > 0.713: **cumple**). |
| 2.2 Red híbrida + riesgo discreto | Elegida `seq=180, tab=1, head=hazard`: AUC OOF 0.827 / 0.793 / 0.778 (criterio ≥ 0.76: **cumple**); holdout 0.823 / 0.798 / 0.782, al nivel del LightGBM. Peso en el stacking 0.05 → 0.23–0.41, pero el stack casi no mejora (ΔAUC OOF +0.000 a +0.003, dentro del ruido). |
| 2.3 Pre-entrenamiento | Mejora OOF fuera del ruido (+0.010 a +0.017) pero **no** en la validación temporal (IC de ΔAUC incluye 0): **no se adopta**; queda con `pre=1`. |
| 2.4 Cierre | Pipeline completo re-corrido, dashboard probado con `AppTest` (4 vistas OK), documentación actualizada. |
| Extra | `src/data.py` ahora es determinista (antes ~0.1 % de los días variaba según la cantidad de hilos de DuckDB). |

## ¿El límite son los datos? (hecho, `python -m src.diagnose temporal|learning|retrain|drift`)

- **Caída temporal (0.78 → 0.64) = ~40 % falta de eventos, ~60 % deriva.** Recortar filas no cuesta nada; limitar a los
  66 vehículos con evento que había en T cuesta ~0.05; el período, ~0.08 más. (Una primera versión del control igualaba
  solo filas y concluía erróneamente que era todo deriva.)
- **Curva de aprendizaje**: ~+0.03 AUC por duplicación de vehículos, sin aplanarse.
- **Deriva**: las features cambian (adversarial AUC 0.88) pero quitarlas no ayuda; cambia la relación con el evento a
  medida que la flota envejece (eventos por 100 vehículos: 1.3 → 8.6 por trimestre).
- **Reentreno mensual**: +0.035 a +0.038 AUC en la flota monitoreada; ≈0 en vehículos nuevos.
- Detalle: README, sección "¿Qué limita el desempeño?"; salidas `data/{temporal_decomp,learning_curve,retrain,drift}.json`.

## Próximos pasos sugeridos

1. Reentreno mensual en producción para la flota monitoreada; ver si pesar los datos recientes o la historia propia del
   vehículo mejora a los vehículos nuevos.
2. Conseguir más vehículos con evento (la palanca con mayor retorno medido) y, si es posible, fechas de inicio de
   síntomas en lugar de fechas de identificación en taller.

## Pendiente

- Traer a pat0top `models/` y `data/` si se va a presentar desde ahí (`scp -r pcpat0:~/Projects/uade-powertrain/{data,models} ...`).
  Los resultados viejos de pat0top quedaron en `data/pat0top/` en pcpat0.
- Opcional: entrenamiento adversarial contra país/cohorte (*gradient reversal*) y modelo jerárquico viajes → días
  (`docs/DOCUMENTACION_TECNICA.md` §13).

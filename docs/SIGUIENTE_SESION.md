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

## ¿El límite son los datos? (hecho, `python -m src.diagnose temporal|learning`)

- **Caída temporal = deriva.** Con el 28 % de las filas el LightGBM rinde igual; el período posterior a T se predice
  bien (0.767) si el modelo lo vio, y cae a 0.653 si no (ΔAUC −0.116 [−0.19, −0.04]). No se sabe todavía si la deriva
  es física (estación, envejecimiento) o del proceso de etiquetado.
- **Curva de aprendizaje**: ~+0.03 AUC por duplicación de vehículos, sin aplanarse (LightGBM 0.742 / 0.771 / 0.788 /
  0.800 con 25 / 50 / 75 / 100 %). Días por vehículo no aportan.
- Detalle: README, sección "¿Qué limita el desempeño?"; salidas en `data/temporal_decomp.json` y
  `data/learning_curve.json`.

## Próximos pasos sugeridos

1. **Frecuencia de reentrenamiento**: simular despliegue con reentrenos mensuales / trimestrales después de T y medir
   cuánto se recupera del 0.653.
2. **Origen de la deriva**: validación adversarial (clasificar días antes vs después de T) para ver qué features
   cambian, y revisar si la tasa y el calendario de identificación de eventos cambian en el tiempo.

## Pendiente

- Traer a pat0top `models/` y `data/` si se va a presentar desde ahí (`scp -r pcpat0:~/Projects/uade-powertrain/{data,models} ...`).
  Los resultados viejos de pat0top quedaron en `data/pat0top/` en pcpat0.
- Opcional: entrenamiento adversarial contra país/cohorte (*gradient reversal*) y modelo jerárquico viajes → días
  (`docs/DOCUMENTACION_TECNICA.md` §13).

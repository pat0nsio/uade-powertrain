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

## Próxima sesión: ¿el límite son los datos?

Todos los modelos convergen a ~0.78–0.80 AUC 90 d (OOF/holdout) y a ~0.65 en la validación temporal; combinarlos no
suma. Hipótesis: el techo lo ponen los datos (pocos eventos, etiquetas = fecha de identificación en taller, sin presión
diferencial del DPF), no el modelo. Dos pruebas para medirlo en vez de inferirlo (~20 min cada una con GPU):

1. **Curva de aprendizaje**: entrenar LightGBM y la red con 25 / 50 / 75 / 100 % de los vehículos de train (mismos
   folds, submuestreo por vehículo estratificado por fallado; varias repeticiones por fracción) y graficar AUC/AP OOF
   con IC. Si sigue subiendo en 100 %, más vehículos ayudarían; si ya se aplanó, el límite es calidad de etiquetas o
   señales.
2. **Descomponer la caída temporal** (0.78 → 0.65): entrenar el modelo del holdout por vehículo con la misma cantidad
   de datos (filas/vehículos/eventos) que había antes de T = 2026-01-01. Si cae igual, es falta de datos; si no, es
   deriva (estacionalidad, envejecimiento de la flota, corte de la bandera de regeneración) y la respuesta es
   reentrenar seguido.

Mismas reglas: decidir solo con OOF o con el período previo a T, reportar IC por vehículo.

## Pendiente

- Traer a pat0top `models/` y `data/` si se va a presentar desde ahí (`scp -r pcpat0:~/Projects/uade-powertrain/{data,models} ...`).
  Los resultados viejos de pat0top quedaron en `data/pat0top/` en pcpat0.
- Opcional: entrenamiento adversarial contra país/cohorte (*gradient reversal*) y modelo jerárquico viajes → días
  (`docs/DOCUMENTACION_TECNICA.md` §13).

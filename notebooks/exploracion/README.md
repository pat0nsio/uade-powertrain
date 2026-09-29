# Exploración en notebooks

Proceso de exploración que se hizo en paralelo al pipeline de `src/`, paso a paso y con controles numéricos en cada
etapa. No reemplaza al pipeline: sirve como validación independiente y como registro de cómo se llegó a varias
decisiones. Varios hallazgos coinciden con los de `src/` (etiquetas v1 inválidas, dos orígenes de fecha de producción,
sesgo de cohorte, vehículos con más de un evento); otros son aportes nuevos (ver abajo).

## Cómo correrlo

Desde esta carpeta, con los CSV de Ford en `Datasets/` (misma estructura que usa `src/data.py`):

```bash
pip install jupyterlab duckdb pandas pyarrow lightgbm scikit-learn
jupyter lab            # correr en orden: 01 → 02 → 03 → 04
```

Todo lo que se genera va a `salidas/` (no versionado): `fic.duckdb`, `dataset.parquet`, `altitud_ciudades.csv`,
`scores_dashboard.*`. DuckDB usa límite de 4 GB y 4 hilos. Al terminar cada notebook se cierra la conexión
(`con.close()`), porque DuckDB no deja abrir la misma base desde dos procesos.

| Notebook | Qué hace | Control principal |
|---|---|---|
| `01_organizar_datos` | Carga, alineación temporal, filtro desde la venta, calidad, viajes limpios y ralentí | 277 fallados, 727 sanos (709 con venta); 1.611.699 viajes válidos |
| `02_agregacion_diaria` | Una fila por vehículo-día, con variables del mecanismo de regeneración | 101.564 días de fallados |
| `03_features_y_etiqueta` | Ventanas de 14/30 días, tendencias, etiqueta a 30 días, `dataset.parquet` | 5.719 positivos de 271 fallados |
| `04_modelo` | LightGBM con `GroupKFold` por vehículo, experimentos, SHAP, exportación | PR-AUC 0,176 (azar 0,028) |

## Hallazgos que aporta esta exploración

1. **Período antes de la venta.** La telemetría arranca en la producción. El 8,3 % de las filas es uso en planta,
   transporte y concesionario: 95 % de viajes cortos y ~150 km por mes, contra ~50 % y ~2.000 km del cliente. Además,
   18 sanos nunca se vendieron. Acá se descarta todo lo anterior a `prod + daysUntilSale`. **`src/` hoy no aplica este
   filtro**, y afecta más a ventanas de 90 días y a las tasas de vida útil.
2. **Ralentí como minutos.** Los viajes de 0 km con duración (motor encendido sin moverse) acumulados en 30 días
   salieron como la segunda variable más importante en SHAP. Antes del evento: 217 contra 174 minutos por mes.
3. **Mecanismo de regeneración, viaje a viaje.** Una regeneración completa (hollín de ~96 % a ~41 %) dura en promedio
   58 minutos de viaje; las interrumpidas ocurren en viajes de 9 a 22 minutos. Antes del evento hay menos regeneraciones
   completas (5,1 contra 6,5 por mes). Sumar estas variables subió el PR-AUC de 0,139 a 0,159–0,176.
4. **País contra altitud.** El país mejora el modelo, pero la tasa de falla por país depende de cómo se armó la muestra
   (Chile 46 %, Argentina 8 %). Reemplazarlo por la **altitud de la ciudad de venta con restricción monotónica** recupera
   parte de la mejora con una variable física y aplicable a cualquier país. Con la altitud libre, parte de la mejora
   era el modelo reconociendo ciudades.
5. **Viajes cortos.** Dentro de los fallados, los viajes cortos no aumentan antes del evento y en SHAP bajan el riesgo.
   Lo que sí cambia es el ralentí y las regeneraciones que no se completan.
6. **Chequeo de cohorte.** Además de las falsas alarmas en sanos, conviene medir cuánto alerta el modelo en fallados
   cuando el evento está lejos (> 90 días): si alerta casi tanto como cerca del evento, reconoce "autos que fallan" en
   vez de detectar degradación. En esta exploración dio 7 % (sanos: 5 %).

## Limitaciones conocidas de esta exploración

Resueltas en `src/`, no acá:
- No deduplica los ~40 k viajes repetidos.
- Usa la bandera `Regenerations` tal cual. Esa señal se corta el 26/05/2026 y venía perdiendo eventos desde marzo, así
  que `regeneraciones_30d` y `dist_regen_30d` están afectadas. Las variables del mecanismo, que salen de los viajes, no.
- Deja como sanos a 12 vehículos que figuran como fallados solo en v1 (`src/` saca 27 conflictos; acá, 15).
- Un solo evento por vehículo (el primero) y un solo horizonte (30 días); sin holdout ni validación temporal.

## Otros archivos

- `ciudades_coordenadas.csv`: coordenadas de las 153 ciudades de venta (142 de GeoNames y 10 cargadas a mano; "CAPITAL",
  de Argentina, sin identificar). La altitud se descarga de la API de elevación de Open-Meteo en el notebook 04.

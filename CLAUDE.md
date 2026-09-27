# DPF Health Copilot — instrucciones para Claude Code

Proyecto para el Ford Innovation Challenge III: predicción temprana de degradación del filtro de partículas (DPF) a
partir de telemetría. Arquitectura y algoritmos: `docs/DOCUMENTACION_TECNICA.md`. Resultados y cómo correrlo: `README.md`.

Estado de la última sesión (GPU en `pcpat0`, hecha) y pendientes: `docs/SIGUIENTE_SESION.md`.

## Reglas del proyecto

- **Commits sin coautoría**: nunca agregar `Co-Authored-By` ni atribución a Claude. Autor:
  `git -c user.name="pat0nsio" -c user.email="patricioameri@gmail.com" commit ...` si git no tiene identidad configurada.
- **DuckDB siempre con límite de recursos** (`config.local.json`, no versionado; ver `src/config.py`). Sin límite, una
  consulta sobre los ~10 M de eventos congeló una máquina de 16 GB. Nada de subconsultas `EXISTS` correlacionadas sobre
  tablas grandes: usar funciones de ventana o `numpy.searchsorted` por vehículo.
- **El código no se acopla a la GPU**: dispositivo, precisión mixta, hilos y parámetros de entrenamiento salen de
  `config.local.json`; todo tiene que seguir corriendo en CPU.
- **El holdout (fold = −1) nunca se usa para decidir** (hiperparámetros, umbrales, arquitectura). Toda elección se hace
  con predicciones OOF de los folds de train o, en la validación temporal, con el período de calibración previo a T.
- **Nada de información del futuro**: las features solo miran hacia atrás (hay un test en `python -m src.features`), y
  "sano" en calibración significa "sin evento conocido a esa fecha", no la etiqueta final.
- Edad, odómetro y fecha **no son features** (sesgo de cohorte: los fallados son más viejos).
- Reportar resultados con honestidad: IC por vehículo, validación temporal, y decir cuándo una mejora está dentro del ruido.
- Jobs largos en `tmux` o en segundo plano, con log en `data/*.log`.
- `Datasets/` es confidencial (Ford): no subirlo a ningún servicio externo ni commitearlo.

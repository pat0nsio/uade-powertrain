# **Ford Innovation Challenge III AI Edition** 

**Detalle de Desafío** 

**Área: Data-Driven Powertrain Intelligence** 

Mentor 1: Camila Domínguez Zandoná - cdomin36@ford.com 

Mentor 2: Nicolás Gagliardi - ngaglia2@ford.com 

SPOC: 

Fecha: 11/09/2026 

## **Índice** 

|**Índice**||**1**|
|---|---|---|
|**1. Des**|**cripción General del Desafío**|**2**|
|1.1.|Título del Desafío . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>2|
|1.2.|Contexto<br>. . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>2|
|1.3.|Desafío . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>2|
|**2. Con**|**texto del Proceso**|**3**|
|2.1.|Descripción del Proceso Actual . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>3|
|2.2.|Ubicación del Flujo . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>3|
|**3. Defi**|**inición del Desafío**|**4**|
|3.1.|Desafío Específico . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>4|
|3.2.|Impacto Operativo . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>4|
|**4. Dat**|**os Técnicos Relevantes**|**5**|
|4.1.|Equipos o Sistemas Existentes . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>5|
|4.2.|Restricciones Técnicas . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>5|
|**5. Exp**|**ectativas de Solución**|**6**|
|5.1.|Objetivos de Mejora . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>6|
|5.2.|Criterios de Evaluación<br>. . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>6|
|**6. Info**|**rmación Complementaria**|**7**|
|6.1.|Documentación Disponible . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>7|
|6.2.|Referentes Técnicos . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>7|
|**7. Ane**|**xo**|**8**|
|7.1.|Tabla _Trip Summary_ . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>8|
|7.2.|Tabla _DynamicInformation_<br>. . . . . . . . . . . . . . . .|. . . . . . . . . . . .<br>9|
|7.3.|Tabla _Vehicle General Information_ . . . . . . . . . . . .|. . . . . . . . . . . .<br>10|



1 

## **1. Descripción General del Desafío** 

### **1.1 Título del Desafío** 

Metodología de Inteligencia Artificial para la predicción temprana de eventos de degradación en la eficiencia de combustión e ingreso de oxígeno al motor, a partir de datos de vehículos conectados. 

### **1.2 Contexto** 

En Ford ponemos la experiencia del cliente en el centro de nuestros esfuerzos, es por ello que buscamos mejorar siempre la eficiencia, el consumo y la comodidad. Con este fin, utilizaremos datos de vehículos conectados, que nos permiten entender cómo nuestros clientes usan nuestros productos, y brindarles la mejor experiencia posible. 

### **1.3 Desafío** 

Te invitamos a desarrollar una metodología o herramienta innovadora, basada en Inteligencia Artificial, Machine Learning y/o Deep Learning, capaz de predecir eventos de degradación debido al uso indebido del cliente. Utilizando datos de vehículos conectados que nos permiten identificar automáticamente patrones y correlaciones entre los datos generados, logramos entender cómo nuestros clientes usan nuestros productos, permitiendo así, anticiparnos al evento de disminución de eficiencia, habilitar mantenimientos predictivos/preventivos y mejorar significativamente la experiencia del cliente. 

2 

## **2. Contexto del Proceso** 

### **2.1 Descripción del Proceso Actual** 

El vehículo cuenta con distintos tipos de sensores que monitorean de forma continua el comportamiento del mismo, reportando datos a través de telemetría del vehículo conectada hacia una base de datos en la nube. El vehículo tiene definidos Triggers que inician la comunicación entre módulos para la recolección de datos y el armado y envío de paquetes a la nube. 

Estos datos se generan de manera continua durante el uso normal del vehículo por parte del cliente. Hoy el análisis es reactivo, ya que la degradación se identifica una vez manifestada, ya sea en un service o cuando el cliente reporta dicha pérdida, no de forma anticipada. 

El uso indebido del cliente, como trayectos predominantemente cortos, operación sostenida en condiciones que no permiten completas los ciclos naturales de recuperación del sistema, patrones de conducción urbana intensiva, y otros, generan una acumulación progresiva que deteriora la eficiencia de combustión a los largo del tiempo. 

### **2.2 Ubicación del Flujo** 

El desafío se ubica en la etapa de post-venta, cuando el cliente ya tiene su vehículo, apoyándose en los datos generados por telemetría remota del vehículo conectado, a diferencia de un control del proceso de planta. La detección temprana impacta directamente en la experiencia de servicio (mantenimiento predictivo) y en la percepción de calidad por parte del cliente. 

3 

## **3. Definición del Desafío** 

### **3.1 Desafío Específico** 

Nuestros vehículos generan de forma continua datos operativos a través de la conectividad embarcada. Hoy, la identificación de una degradación en la eficiencia de combustión o en el ingreso de oxígeno al motor ocurre de manera tardía. Se requiere desarrollar un modelo predictivo capaz de anticipar, a partir de patrones históricos de uso y variables de motor, el momento en que un vehículo se aproxima a un evento de degradación, permitiendo habilitar acciones de mantenimiento predictivo/preventivo antes de que el cliente perciba el problema. 

### **3.2 Impacto Operativo** 

No anticipar estos eventos impacta negativamente en la experiencia y satisfacción del cliente, incrementa los costos de garantía y reparación, reduce la vida útil percibida del vehículo, y genera ineficiencias en la planificación de mantenimiento de la red de concesionarios. 

4 

## **4. Datos Técnicos Relevantes** 

Se van a entregar distintas tablas: 

- _Trip Summary_ : tiene información sobre distintos estados del vehículo al comenzar y finalizar el viaje. Para más información ver el anexo 7.1. 

- _DynamicInformation_ : tiene información sobre distintos estados del vehículo en todo momemento que este envía datos. Para más información ver el anexo 7.2. 

- _Vehicle General Information_ : tiene información general sobre el vehículo. Para más información ver el anexo 7.3. 

### **4.1 Equipos o Sistemas Existentes** 

Se dispondrá de un histórico de recolección por vehículo para el entrenamiento del modelo, tanto de vehículos con degradación de eficiencia como de los que no presentaron; periodicidad y volumen a confirmar por el responsable del área. 

### **4.2 Restricciones Técnicas** 

El dataset entregado a los participantes contendrá variables renombradas y, de corresponder, anonimizadas (identificador de vehículo), a fin de preservar la confidencialidad de los sistemas involucrados. 

5 

## **5. Expectativas de Solución** 

Se espera que toda solución tenga algún aspecto de por lo menos uno de los siguientes campos de la innovación: análisis de datos, inteligencia artificial, machine learning, deep learning. 

### **5.1 Objetivos de Mejora** 

Anticipar, con la mayor cantidad de tiempo/kilometraje posible antes del evento, la degradación de eficiencia de combustión, habilitando alertas de mantenimiento predictivo/preventivo. 

### **5.2 Criterios de Evaluación** 

Qué se valorará: viabilidad técnica, costo, carácter innovador, calidad de tratamiento de datos (limpieza y preparación de datos, manejo de valores faltantes/atípicos, consistencia y trazabilidad del pipeline), claridad en la forma de mostrar los datos (visualizaciones, dashboards, gráficos de tendencia, etc.) explicabilidad del modelo (interpretabilidad de resultados), power pitch al vender la idea. 

6 

## **6. Información Complementaria** 

### **6.1 Documentación Disponible** 

Adjunta en el .ZIP (dataset de telemetría, diccionario de variables). 

### **6.2 Referentes Técnicos** 

Mentora 1: Camila Domíguez Zandoná – cdomin36@ford.com Mentor 2: Nicolás Gagliardi – ngaglia2@ford.com 

7 

## **7. Anexo** 

### **7.1 Tabla** **_Trip Summary_** 

Columnas: 

1. Vin: identificador único del vehículo (VIN). 

2. TripDatetimeStart: fecha y hora de inicio del viaje. 

3. TripDatetimeEnd: fecha y hora de finalización del viaje. 

4. OdometerTripStart: kilometraje del odómetro al inicio del viaje [km]. 

5. OdometerTripEnd: kilometraje del odómetro al final del viaje [km]. 

6. KilometerPerHour: velocidad promedio del vehículo durante el viaje [km/h]. 

7. FuelLvlStartPc: nivel de combustible al inicio del viaje, expresado como porcentaje del tanque [ %]. 

8. FuelLvlEndPc: nivel de combustible al final del viaje, expresado como porcentaje del tanque [ %]. 

9. FuelLvlAutonomyStart: autonomía estimada del vehículo según el nivel de combustible al inicio del viaje [km]. 

10. EngineOilLifePCStart: porcentaje de vida útil restante del aceite de motor al inicio del viaje [ %]. 

11. EngineOilLifePCEnd: porcentaje de vida útil restante del aceite de motor al final del viaje [ %]. 

12. EngineTemperatureMin: temperatura mínima registrada del motor durante el viaje [°C]. 

13. EngineTemperatureMax: temperatura máxima registrada del motor durante el viaje [°C]. 

14. EngineTemperatureAvg: temperatura promedio del motor durante el viaje [°C]. 

15. CoolantTemperatureStart: temperatura del refrigerante del motor al inicio del viaje [°C]. 

16. CoolantTemperatureEnd: temperatura del refrigerante del motor al final del viaje [°C]. 

17. DieselParticulateFilterStart: nivel de saturación/carga de hollín del filtro de partículas diésel (DPF) al inicio del viaje [ %]. 

18. DieselParticulateFilterEnd: nivel de saturación/carga de hollín del filtro de partículas diésel (DPF) al final del viaje [ %]. 

19. VehicleGPSLatDataStart: coordenada de latitud GPS del vehículo al inicio del viaje [grados decimales]. 

20. VehicleGPSLatDataEnd: coordenada de latitud GPS del vehículo al final del viaje [grados decimales]. 

21. VehicleGPSLongDataStart: coordenada de longitud GPS del vehículo al inicio del viaje [grados decimales]. 

22. VehicleGPSLongDataEnd: coordenada de longitud GPS del vehículo al final del viaje [grados decimales]. 

8 

23. VehicleElevationRangeStart: elevación/altitud del vehículo respecto al nivel del mar al inicio del viaje [m]. 

24. VehicleElevationRangeEnd: elevación/altitud del vehículo respecto al nivel del mar al final del viaje [m]. 

25. TirePressureLFStart: presión de la llanta delantera izquierda (Left Front) al inicio del viaje [PSI]. 

26. TirePressureLFEnd: presión de la llanta delantera izquierda (Left Front) al final del viaje [PSI]. 

27. TirePressureRFStart: presión de la llanta delantera derecha (Right Front) al inicio del viaje [PSI]. 

28. TirePressureRFEnd: presión de la llanta delantera derecha (Right Front) al final del viaje [PSI]. 

29. TirePressureLRStart: presión de la llanta trasera izquierda (Left Rear) al inicio del viaje [PSI]. 

30. TirePressureLREnd: presión de la llanta trasera izquierda (Left Rear) al final del viaje [PSI]. 

31. TirePressureRRStart: presión de la llanta trasera derecha (Right Rear) al inicio del viaje [PSI]. 

32. TirePressureRREnd: presión de la llanta trasera derecha (Right Rear) al final del viaje [PSI]. 

33. AirTemperatureStart: temperatura ambiente exterior registrada al inicio del viaje [°C]. 

34. AirTemperatureEnd: temperatura ambiente exterior registrada al final del viaje [°C]. 

35. AirTemperatureMin: temperatura ambiente mínima registrada durante el viaje [°C]. 

36. AirTemperatureMax: temperatura ambiente máxima registrada durante el viaje [°C]. 

37. AirTemperatureAvg: temperatura ambiente promedio durante el viaje [°C]. 

38. ManualRegenerationSootStart: nivel de hollín acumulado en el sistema antes de una regeneración manual del DPF al inicio del viaje [ %]. 

39. ManualRegenerationSootEnd: nivel de hollín acumulado en el sistema después de una regeneración manual del DPF al final del viaje [ %]. 

40. TripNumber: número identificador/secuencial del viaje asociado al vehículo. 

### **7.2 Tabla** **_DynamicInformation_** 

1. VehicleCode: identificador del vehículo. 

2. EventTimestamp: timestamp del momento de las mediciones. 

3. OdometerValue: kilometraje total del vehículo [km]. 

4. Acumulation: acumulación en el filtro de aire [ %]. 

5. Message: mensaje de la ecu del proceso actual. 

6. Regenerations: si está regenerando al momento de la medición [bool]. 

7. DistanceBetweenRegenerations: distancia recorrida entre regeneraciones en el filtro de aire. 

9 

### **7.3 Tabla** **_Vehicle General Information_** 

VehCode: código de identificación del vehículo. 

- IdentificationDate: días desde producción en el que se identificó la pérdida de eficiencia. 

- daysUntilSale: días desde la producción hasta que se vendió la unidad. 

- ProductionDay: días desde el primer vehículo producido de las listas provistas. 

- Engine: motor del vehículo. 

- ModelSeries: serie del catálogo. 

- SalesCountryCd: país en el que se vendió la unidad. 

10 


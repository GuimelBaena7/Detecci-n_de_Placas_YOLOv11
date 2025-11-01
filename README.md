# 🚗 Detección de Placas con YOLOv11 + OCR

Sistema avanzado de detección y reconocimiento de placas vehiculares utilizando YOLOv11, algoritmo SORT para seguimiento y EasyOCR para lectura de texto.

## ✨ Características

- 🚗 **Detección de vehículos** usando YOLOv11 (coches, motocicletas, autobuses, camiones)
- 🔍 **Detección de placas** con modelo personalizado entrenado
- 📍 **Seguimiento de vehículos** con algoritmo SORT
- 🔤 **Reconocimiento OCR** con EasyOCR (inglés y español)
- 📊 **Interpolación de datos** faltantes para trayectorias suaves
- 🎬 **Visualización avanzada** con video de salida
- 📈 **Estadísticas detalladas** de rendimiento

## 🚀 Instalación Rápida

### Opción 1: Instalación automática (Recomendada)
```bash
python install.py
```

### Opción 2: Instalación manual
```bash
pip install -r requirements.txt
```

## 📋 Requisitos

### Modelos necesarios:
- `yolo11n.pt` - Modelo YOLOv11 (se descarga automáticamente)
- `license_plate_detector.pt` - Modelo personalizado para placas ⚠️ **REQUERIDO**

### Dependencias Python:
- ultralytics >= 8.3.0
- opencv-python
- numpy
- pandas
- scipy
- filterpy
- easyocr

## 🎯 Uso

### 🔥 Pipeline completo (Recomendado)
```bash
python run_all.py
```
Ejecuta todo el proceso: detección → interpolación → visualización

### 📝 Ejecución paso a paso

#### 1. Detección principal
```bash
python main.py
```
- Detecta vehículos y placas
- Lee texto con OCR
- Genera `test.csv`

#### 2. Interpolación de datos
```bash
python add_missing_data.py
```
- Completa trayectorias faltantes
- Genera `test_interpolated.csv`

#### 3. Visualización
```bash
python visualize.py
```
- Crea video con detecciones
- Genera `out.mp4`

## 📁 Estructura del Proyecto

```
Deteccion-de-Placas-YOLOv11/
├── 📄 main.py                    # Script principal de detección
├── 🛠️ util.py                    # Funciones OCR y utilidades
├── 📊 add_missing_data.py        # Interpolación de datos
├── 🎬 visualize.py               # Generación de video
├── 🚀 run_all.py                 # Pipeline completo
├── 📦 install.py                 # Instalador automático
├── 📋 requirements.txt           # Dependencias
├── 🤖 yolo11n.pt                # Modelo YOLOv11
├── 🎯 license_plate_detector.pt  # Modelo de placas
├── 📁 sort/                      # Algoritmo de seguimiento
│   ├── __init__.py
│   └── sort.py
├── 📁 imagenes/                  # Placas recortadas
└── 📄 README.md
```

## 📊 Archivos de Salida

### CSV de resultados (`test.csv`)
```csv
frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score
0,1,[100 200 300 400],[150 250 200 280],0.95,ABC123,0.87
```

### Columnas:
- `frame_nmr`: Número de frame
- `car_id`: ID único del vehículo
- `car_bbox`: Coordenadas del vehículo [x1,y1,x2,y2]
- `license_plate_bbox`: Coordenadas de la placa [x1,y1,x2,y2]
- `license_plate_bbox_score`: Confianza de detección (0-1)
- `license_number`: Texto leído de la placa
- `license_number_score`: Confianza del OCR (0-1)

## 🔧 Configuración Avanzada

### Ajustar parámetros OCR (util.py)
```python
# Cambiar idiomas soportados
reader = easyocr.Reader(['en', 'es', 'fr'], gpu=False)

# Ajustar patrones de placas
patterns = [
    r'^[A-Z]{3}[0-9]{3}$',      # ABC123
    r'^[A-Z]{3}[0-9]{2}[A-Z]$', # ABC12D
    # Agregar más patrones aquí
]
```

### Modificar clases de vehículos (main.py)
```python
# Clases COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
vehicles = [2, 3, 5, 7]  # Agregar/quitar según necesidad
```

## 📈 Rendimiento

### Factores que afectan la precisión:
- ✅ **Calidad del video**: Mayor resolución = mejor OCR
- ✅ **Iluminación**: Buena luz mejora la lectura
- ✅ **Ángulo de la cámara**: Frontal es óptimo
- ✅ **Velocidad del vehículo**: Menor velocidad = mejor captura
- ✅ **Tamaño de la placa**: Placas más grandes se leen mejor

### Optimizaciones incluidas:
- 🔄 **Múltiples intentos OCR** (imagen original + preprocesada)
- 🎯 **Corrección de caracteres** similares (O→0, I→1, etc.)
- 📏 **Redimensionamiento automático** de placas pequeñas
- 🧹 **Filtrado de ruido** con morfología
- 🔍 **Detección de inversión** automática

## 🐛 Solución de Problemas

### Error: "No module named 'easyocr'"
```bash
pip install easyocr
```

### Error: "license_plate_detector.pt not found"
- Asegúrate de tener el modelo personalizado en el directorio raíz
- El modelo debe estar entrenado para detectar placas

### OCR no funciona bien
- Verifica que las placas sean visibles y legibles
- Ajusta los patrones de validación en `license_complies_format()`
- Considera entrenar un modelo OCR personalizado

### Video de salida corrupto
- Verifica que el codec 'mp4v' sea compatible
- Prueba cambiar el codec en `visualize.py`

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [Ultralytics](https://ultralytics.com/) por YOLOv11
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) por el motor OCR
- [SORT](https://github.com/abewley/sort) por el algoritmo de seguimiento

## 📞 Soporte

Si tienes problemas o preguntas:
1. Revisa la sección de solución de problemas
2. Verifica que todos los requisitos estén instalados
3. Ejecuta `python install.py` para verificar la instalación

---

**⭐ Si este proyecto te fue útil, ¡dale una estrella!**
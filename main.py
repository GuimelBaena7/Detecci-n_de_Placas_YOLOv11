from ultralytics import YOLO
import cv2
import numpy as np
import os

from sort.sort import Sort
from util import get_car, write_csv, read_license_plate

# Crear carpeta "imagenes" si no existe
os.makedirs("imagenes", exist_ok=True)

results = {}
mot_tracker = Sort()

# Cargar modelos
coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Cargar video
ruta_video = input("👉 Ingresa la ruta o nombre del archivo de video: ")
cap = cv2.VideoCapture(ruta_video)

# Clases de vehículos en COCO: car, motorcycle, bus, truck
vehicles = [2, 3, 5, 7]

# Procesar frames
frame_nmr = -1
ret = True

print("🚀 Iniciando procesamiento...")

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if ret:
        results[frame_nmr] = {}
        
        # Detectar vehículos
        detections = coco_model(frame)[0]
        detections_ = []
        num_vehiculos = 0
        
        if detections.boxes is not None:
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    num_vehiculos += 1

        print(f"🟩 Frame {frame_nmr}: Vehículos detectados = {num_vehiculos}")

        # Rastrear vehículos
        if len(detections_) == 0:
            detections_ = np.empty((0, 5))
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detectar placas
        license_plates = license_plate_detector(frame)[0]
        num_placas = 0
        
        if license_plates.boxes is not None:
            for license_plate in license_plates.boxes.data.tolist():
                num_placas += 1
                x1, y1, x2, y2, score, class_id = license_plate

                # Asignar placa a vehículo
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Recortar placa
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    # Guardar imagen de placa
                    try:
                        nombre_imagen = f"imagenes/placa_frame{frame_nmr}_car{int(car_id)}.jpg"
                        cv2.imwrite(nombre_imagen, license_plate_crop)
                        print(f"💾 Placa guardada: {nombre_imagen}")
                    except Exception as e:
                        print(f"⚠️ Error al guardar imagen: {e}")

                    # Leer texto de placa con OCR
                    license_text, text_score = read_license_plate(license_plate_crop)

                    # Guardar resultados
                    if license_text is not None:
                        results[frame_nmr][int(car_id)] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_text,
                                'bbox_score': score,
                                'text_score': text_score
                            }
                        }
                        print(f"✅ Placa leída: {license_text} (Confianza: {text_score:.2f})")
                    else:
                        results[frame_nmr][int(car_id)] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': 'UNKNOWN',
                                'bbox_score': score,
                                'text_score': 0.0
                            }
                        }
                        print(f"⚠️ No se pudo leer placa en Frame {frame_nmr}, Car ID {int(car_id)}")

        print(f"🟦 Frame {frame_nmr}: Placas detectadas = {num_placas}")

# Estadísticas finales
print(f"\n📊 Procesamiento completado")
print(f"📈 Total de frames procesados: {frame_nmr + 1}")

total_detections = sum(len(results[f]) for f in results)
print(f"📋 Total de detecciones: {total_detections}")

# Contar placas leídas exitosamente
ocr_success = 0
for frame in results.values():
    for car_data in frame.values():
        if 'license_plate' in car_data:
            text = car_data['license_plate']['text']
            if text not in ['UNKNOWN', 'NO_OCR', '']:
                ocr_success += 1

if total_detections > 0:
    success_rate = (ocr_success / total_detections) * 100
    print(f"🔤 Placas leídas exitosamente: {ocr_success}/{total_detections} ({success_rate:.1f}%)")

# Guardar resultados
write_csv(results, './test.csv')
print(f"✅ Archivo CSV guardado: ./test.csv")

cap.release()
print("🎉 Procesamiento completado exitosamente")
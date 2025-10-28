from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, write_csv
# from util import read_license_plate  # OCR deshabilitado
# Crear carpeta "imagenes" si no existe
import os
os.makedirs("imagenes", exist_ok=True)

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolo11n.pt')  # YOLOv11 nano model
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
ruta_video = input(" Ingresa la ruta o nombre del archivo de video: ")
cap = cv2.VideoCapture(ruta_video)

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        num_vehiculos = 0
        if detections.boxes is not None:
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    num_vehiculos += 1

        print(f" Frame {frame_nmr}: Vehículos detectados = {num_vehiculos}")

        # track vehicles
        if len(detections_) == 0:
            detections_ = np.empty((0, 5))
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        num_placas = 0
        if license_plates.boxes is not None:
            for license_plate in license_plates.boxes.data.tolist():
                num_placas += 1
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    
                    # Guardar la imagen recortada de la placa
                    try:
                        nombre_imagen = f"imagenes/placa_frame{frame_nmr}_car{int(car_id)}.jpg"
                        cv2.imwrite(nombre_imagen, license_plate_crop)
                        print(f" Placa guardada: {nombre_imagen}")
                    except Exception as e:
                        print(f" Error al guardar imagen de placa: {e}")

                    # Guardar solo detección sin OCR
                    results[frame_nmr][int(car_id)] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': 'NO_OCR',
                            'bbox_score': score,
                            'text_score': 0.0
                        }
                    }
                    print(f" Datos guardados: Frame {frame_nmr}, Car ID {int(car_id)}")
        print(f" Frame {frame_nmr}: Placas detectadas = {num_placas}")

# write results
print(f"\n Procesamiento completado. Guardando resultados...")
print(f" Total de frames procesados: {frame_nmr + 1}")
print(f" Total de detecciones guardadas: {sum(len(results[f]) for f in results)}")
write_csv(results, './test.csv')
print(f" Archivo CSV guardado: ./test.csv")
cap.release()

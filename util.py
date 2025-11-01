import string
import easyocr
import cv2
import numpy as np
import re
import csv

# Inicializar EasyOCR
reader = easyocr.Reader(['en', 'es'], gpu=True)

# Mapeos de caracteres comunes en placas
dict_char_to_int = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'G': '6', 'Z': '2'}
dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

def write_csv(results, output_path):
    """Guardar resultados en archivo CSV de forma segura"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'
        ])

        for frame_nmr, cars in results.items():
            for car_id, data in cars.items():
                if 'car' in data and 'license_plate' in data:
                    lp = data['license_plate']

                    # Manejo seguro de valores
                    car_bbox = [
                        float(x) if isinstance(x, (int, float, np.floating)) else 0
                        for x in data['car'].get('bbox', [0, 0, 0, 0])
                    ]
                    lp_bbox = [
                        float(x) if isinstance(x, (int, float, np.floating)) else 0
                        for x in lp.get('bbox', [0, 0, 0, 0])
                    ]

                    lp_bbox_score = float(lp.get('bbox_score', 0)) if lp.get('bbox_score') is not None else 0
                    lp_text_score = float(lp.get('text_score', 0)) if lp.get('text_score') is not None else 0
                    lp_text = lp.get('text', 'UNKNOWN')

                    # Convertir listas a strings compactos
                    car_bbox_str = ' '.join(map(str, car_bbox))
                    lp_bbox_str = ' '.join(map(str, lp_bbox))

                    writer.writerow([
                        frame_nmr, car_id, f"[{car_bbox_str}]",
                        f"[{lp_bbox_str}]", lp_bbox_score,
                        lp_text, lp_text_score
                    ])

def license_complies_format(text):
    """Validar formato de placa vehicular"""
    if len(text) < 5 or len(text) > 7:
        return False
    
    patterns = [
        r'^[A-Z]{3}[0-9]{3}$',      # ABC123
        r'^[A-Z]{3}[0-9]{2}[A-Z]$', # ABC12D
        r'^[A-Z]{2}[0-9]{3}[A-Z]$', # AB123C
        r'^[0-9]{3}[A-Z]{3}$',      # 123ABC
    ]
    
    return any(re.match(pattern, text) for pattern in patterns)

def format_license(text):
    """Corregir caracteres comúnmente mal interpretados"""
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
    formatted = ''
    
    for i, char in enumerate(text):
        # Aplicar correcciones contextuales
        if i < 3:  # Primeras posiciones suelen ser letras
            if char in dict_int_to_char:
                formatted += dict_int_to_char[char]
            else:
                formatted += char
        else:  # Posiciones posteriores pueden ser números
            if char in dict_char_to_int:
                formatted += dict_char_to_int[char]
            else:
                formatted += char
    
    return formatted

def preprocess_plate(plate_img):
    """Preprocesamiento optimizado para OCR"""
    if plate_img is None or plate_img.size == 0:
        return None
    
    # Redimensionar si es muy pequeña
    h, w = plate_img.shape[:2]
    if h < 30 or w < 80:
        scale = max(30/h, 80/w)
        new_w, new_h = int(w * scale), int(h * scale)
        plate_img = cv2.resize(plate_img, (new_w, new_h))
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Umbral adaptativo
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )
    
    # Limpieza de ruido
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Detección automática de inversión
    white_ratio = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])
    if white_ratio < 0.4:
        thresh = cv2.bitwise_not(thresh)
    
    return thresh

def read_license_plate(license_plate_crop):
    """Lectura OCR mejorada con múltiples intentos"""
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None
    
    try:
        results = []
        
        # Intento 1: Imagen original
        detections = reader.readtext(license_plate_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for detection in detections:
            _, text, score = detection
            if len(text) >= 4:
                results.append((text, score))
        
        # Intento 2: Imagen preprocesada
        preprocessed = preprocess_plate(license_plate_crop)
        if preprocessed is not None:
            detections = reader.readtext(preprocessed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for detection in detections:
                _, text, score = detection
                if len(text) >= 4:
                    results.append((text, score * 0.9))
        
        # Procesar resultados
        best_text = None
        best_score = 0
        
        for text, score in results:
            text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
            
            # Aplicar correcciones
            formatted = format_license(text)
            
            # Verificar formato y seleccionar el mejor
            if license_complies_format(formatted) and score > best_score:
                best_text, best_score = formatted, score
            elif len(formatted) >= 5 and score > best_score and best_text is None:
                best_text, best_score = formatted, score
        
        return best_text, best_score
        
    except Exception as e:
        print(f"⚠️ Error en OCR: {e}")
        return None, None

def get_car(license_plate, vehicle_track_ids):
    """Asignar placa a vehículo"""
    x1, y1, x2, y2, score, class_id = license_plate

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Si la placa está dentro de la caja del carro
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1

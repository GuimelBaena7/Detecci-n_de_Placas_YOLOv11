import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """Dibujar bordes estilizados alrededor de vehículos"""
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Esquinas superiores
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Esquinas inferiores
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def parse_bbox(bbox_str):
    """Parsear string de coordenadas a lista de números"""
    try:
        return ast.literal_eval(bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    except:
        # Fallback para formatos alternativos
        coords = bbox_str.strip('[]').split()
        return [float(x) for x in coords if x]

def main():
    print("🎬 Iniciando visualización...")
    
    # Cargar datos
    try:
        results = pd.read_csv('./test_interpolated.csv')
        print("📊 Usando datos interpolados")
    except FileNotFoundError:
        try:
            results = pd.read_csv('./test.csv')
            print("📊 Usando datos originales")
        except FileNotFoundError:
            print("❌ No se encontró archivo de datos (test.csv o test_interpolated.csv)")
            return

    # Cargar video
    ruta_video = input("👉 Ingresa la ruta del video: ")
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print("❌ Error al abrir el video")
        return

    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

    print(f"📹 Video: {width}x{height} @ {fps} FPS")

    # Preparar datos de placas por vehículo
    license_plate_data = {}
    
    for car_id in results['car_id'].unique():
        car_data = results[results['car_id'] == car_id]
        
        # Encontrar la mejor lectura de placa para este vehículo
        best_score = 0
        best_text = 'UNKNOWN'
        best_frame = car_data['frame_nmr'].iloc[0]
        
        for _, row in car_data.iterrows():
            if row['license_number_score'] > best_score and row['license_number'] not in ['UNKNOWN', 'NO_OCR', '']:
                best_score = row['license_number_score']
                best_text = row['license_number']
                best_frame = row['frame_nmr']
        
        # Obtener imagen de la placa
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
        ret, frame = cap.read()
        
        license_crop = None
        if ret:
            try:
                bbox = parse_bbox(car_data[car_data['frame_nmr'] == best_frame]['license_plate_bbox'].iloc[0])
                x1, y1, x2, y2 = bbox
                
                if x2 > x1 and y2 > y1:
                    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    if license_crop.size > 0:
                        # Redimensionar para visualización
                        aspect_ratio = (x2 - x1) / (y2 - y1)
                        new_height = 80
                        new_width = int(new_height * aspect_ratio)
                        license_crop = cv2.resize(license_crop, (new_width, new_height))
            except Exception as e:
                print(f"⚠️ Error procesando placa del vehículo {car_id}: {e}")
        
        # Crear imagen placeholder si no hay crop válido
        if license_crop is None or license_crop.size == 0:
            license_crop = np.ones((80, 200, 3), dtype=np.uint8) * 128
            cv2.putText(license_crop, 'NO IMAGE', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        license_plate_data[car_id] = {
            'text': best_text,
            'crop': license_crop,
            'score': best_score
        }

    print(f"🚗 Procesados {len(license_plate_data)} vehículos únicos")

    # Procesar video frame por frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Obtener detecciones para este frame
        frame_data = results[results['frame_nmr'] == frame_nmr]
        
        for _, row in frame_data.iterrows():
            try:
                car_id = row['car_id']
                
                # Dibujar vehículo
                car_bbox = parse_bbox(row['car_bbox'])
                car_x1, car_y1, car_x2, car_y2 = car_bbox
                
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), 
                           (0, 255, 0), 15, line_length_x=100, line_length_y=100)
                
                # Dibujar placa
                lp_bbox = parse_bbox(row['license_plate_bbox'])
                lp_x1, lp_y1, lp_x2, lp_y2 = lp_bbox
                
                cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 3)
                
                # Mostrar información de la placa
                if car_id in license_plate_data:
                    plate_info = license_plate_data[car_id]
                    license_crop = plate_info['crop']
                    license_text = plate_info['text']
                    
                    # Posición para mostrar la placa ampliada
                    crop_h, crop_w = license_crop.shape[:2]
                    
                    # Calcular posición arriba del vehículo
                    display_x = max(0, int((car_x1 + car_x2 - crop_w) / 2))
                    display_y = max(crop_h + 60, int(car_y1) - crop_h - 60)
                    
                    # Asegurar que no se salga de la imagen
                    if display_x + crop_w > width:
                        display_x = width - crop_w
                    if display_y < 0:
                        display_y = int(car_y2) + 20
                    
                    # Mostrar imagen de placa ampliada
                    try:
                        frame[display_y:display_y + crop_h, display_x:display_x + crop_w] = license_crop
                        
                        # Fondo para el texto
                        text_bg_y = display_y - 50
                        if text_bg_y < 0:
                            text_bg_y = display_y + crop_h + 10
                        
                        cv2.rectangle(frame, (display_x, text_bg_y), 
                                    (display_x + crop_w, text_bg_y + 40), (0, 0, 0), -1)
                        
                        # Texto de la placa
                        font_scale = min(1.2, crop_w / 150)
                        cv2.putText(frame, license_text, (display_x + 5, text_bg_y + 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
                        
                        # Mostrar confianza si es válida
                        if plate_info['score'] > 0:
                            conf_text = f"Conf: {plate_info['score']:.2f}"
                            cv2.putText(frame, conf_text, (display_x + 5, text_bg_y + 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    except Exception as e:
                        # Si hay error mostrando la imagen, solo mostrar el texto
                        cv2.putText(frame, license_text, (int(car_x1), int(car_y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
            except Exception as e:
                print(f"⚠️ Error procesando frame {frame_nmr}: {e}")
                continue
        
        # Escribir frame al video de salida
        out.write(frame)
        
        # Mostrar progreso cada 100 frames
        if frame_nmr % 100 == 0:
            print(f"📹 Procesando frame {frame_nmr}...")
        
        frame_nmr += 1

    # Limpiar recursos
    out.release()
    cap.release()
    
    print(f"✅ Video generado: ./out.mp4")
    print(f"📊 Total de frames procesados: {frame_nmr}")

if __name__ == "__main__":
    main()
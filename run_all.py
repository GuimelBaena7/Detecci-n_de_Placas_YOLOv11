#!/usr/bin/env python3
"""
Script principal para ejecutar todo el pipeline de detección de placas con OCR
"""
import os
import subprocess
import sys
import time

def print_header(title):
    """Imprimir encabezado estilizado"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def run_script(script_name, description):
    """Ejecutar un script y manejar errores"""
    print_header(description)
    
    if not os.path.exists(script_name):
        print(f"❌ Archivo no encontrado: {script_name}")
        return False
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.getcwd())
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} completado exitosamente")
            print(f"⏱️ Tiempo transcurrido: {elapsed_time:.1f} segundos")
            return True
        else:
            print(f"\n❌ Error en {description} (código: {result.returncode})")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Proceso interrumpido por el usuario")
        return False
    except Exception as e:
        print(f"\n⚠️ Error ejecutando {script_name}: {e}")
        return False

def check_requirements():
    """Verificar archivos y dependencias necesarias"""
    print_header("VERIFICACIÓN DE REQUISITOS")
    
    # Verificar archivos necesarios
    required_files = [
        'main.py', 'util.py', 'add_missing_data.py', 'visualize.py',
        'license_plate_detector.pt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return False
    
    # Verificar directorio sort
    if not os.path.exists('sort/sort.py'):
        print("❌ Falta el algoritmo SORT (sort/sort.py)")
        return False
    
    print("✅ Todos los archivos necesarios están presentes")
    
    # Verificar dependencias de Python
    try:
        import ultralytics
        import cv2
        import numpy
        import pandas
        import scipy
        import filterpy
        import easyocr
        print("✅ Todas las dependencias de Python están instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")
        return False

def show_results():
    """Mostrar resumen de archivos generados"""
    print_header("RESUMEN DE RESULTADOS")
    
    files_to_check = [
        ('test.csv', 'Detecciones originales'),
        ('test_interpolated.csv', 'Datos interpolados'),
        ('out.mp4', 'Video con visualizaciones'),
        ('imagenes/', 'Imágenes de placas recortadas')
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   ✅ {description}: {file_path} ({size:,} bytes)")
            else:
                # Es un directorio
                try:
                    count = len([f for f in os.listdir(file_path) if f.endswith('.jpg')])
                    print(f"   ✅ {description}: {count} imágenes guardadas")
                except:
                    print(f"   ✅ {description}: directorio creado")
        else:
            print(f"   ❌ {description}: {file_path} (no generado)")

def main():
    """Función principal del pipeline"""
    print("🚀 PIPELINE DE DETECCIÓN DE PLACAS CON OCR")
    print("🔤 Versión mejorada con lectura de texto")
    
    # Verificar requisitos
    if not check_requirements():
        print("\n❌ No se pueden ejecutar los scripts. Revisa los requisitos.")
        return
    
    total_start_time = time.time()
    
    # Paso 1: Detección principal
    if not run_script('main.py', 'DETECCIÓN DE VEHÍCULOS Y PLACAS CON OCR'):
        print("\n❌ Falló la detección principal. Deteniendo pipeline.")
        return
    
    # Verificar que se generó el CSV
    if not os.path.exists('test.csv'):
        print("\n❌ No se generó el archivo test.csv")
        return
    
    # Paso 2: Interpolación de datos
    if run_script('add_missing_data.py', 'INTERPOLACIÓN DE DATOS FALTANTES'):
        print("✅ Interpolación completada")
    else:
        print("⚠️ Error en interpolación, continuando con datos originales")
    
    # Paso 3: Visualización
    if run_script('visualize.py', 'GENERACIÓN DE VIDEO CON VISUALIZACIONES'):
        print("✅ Visualización completada")
    else:
        print("⚠️ Error en visualización")
    
    # Mostrar resultados
    total_time = time.time() - total_start_time
    show_results()
    
    print_header("PIPELINE COMPLETADO")
    print(f"⏱️ Tiempo total: {total_time:.1f} segundos")
    print("🎉 ¡Procesamiento exitoso!")
    
    # Mostrar estadísticas si existe el CSV
    try:
        import pandas as pd
        df = pd.read_csv('test.csv')
        total_detections = len(df)
        unique_cars = df['car_id'].nunique()
        successful_ocr = len(df[~df['license_number'].isin(['UNKNOWN', 'NO_OCR', ''])])
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   🚗 Vehículos únicos detectados: {unique_cars}")
        print(f"   📋 Total de detecciones: {total_detections}")
        print(f"   🔤 Placas leídas exitosamente: {successful_ocr}")
        if total_detections > 0:
            success_rate = (successful_ocr / total_detections) * 100
            print(f"   📈 Tasa de éxito OCR: {success_rate:.1f}%")
            
    except Exception as e:
        print(f"⚠️ No se pudieron calcular estadísticas: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
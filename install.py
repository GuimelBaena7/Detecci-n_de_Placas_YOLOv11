#!/usr/bin/env python3
"""
Script de instalación automática para el proyecto de detección de placas
"""
import subprocess
import sys
import os

def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    print("📦 Instalando dependencias...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def download_yolo_model():
    """Descargar modelo YOLOv11 si no existe"""
    if not os.path.exists('yolo11n.pt'):
        print("📥 Descargando modelo YOLOv11...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolo11n.pt')  # Esto descarga automáticamente el modelo
            print("✅ Modelo YOLOv11 descargado")
            return True
        except Exception as e:
            print(f"❌ Error descargando modelo YOLOv11: {e}")
            return False
    else:
        print("✅ Modelo YOLOv11 ya existe")
        return True

def check_license_plate_model():
    """Verificar que existe el modelo de detección de placas"""
    if os.path.exists('license_plate_detector.pt'):
        print("✅ Modelo de detección de placas encontrado")
        return True
    else:
        print("⚠️ Modelo de detección de placas no encontrado (license_plate_detector.pt)")
        print("💡 Asegúrate de tener el archivo license_plate_detector.pt en el directorio del proyecto")
        return False

def create_directories():
    """Crear directorios necesarios"""
    directories = ['imagenes', 'sort']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Directorio creado: {directory}")
        else:
            print(f"✅ Directorio ya existe: {directory}")

def verify_installation():
    """Verificar que todo está instalado correctamente"""
    print("\n🔍 Verificando instalación...")
    
    try:
        # Verificar imports principales
        import ultralytics
        import cv2
        import numpy
        import pandas
        import scipy
        import filterpy
        import easyocr
        
        print("✅ Todas las librerías importadas correctamente")
        
        # Verificar versiones importantes
        print(f"   - OpenCV: {cv2.__version__}")
        print(f"   - NumPy: {numpy.__version__}")
        print(f"   - Pandas: {pandas.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("🚀 INSTALADOR DEL PROYECTO DETECCIÓN DE PLACAS")
    print("=" * 50)
    
    success = True
    
    # Paso 1: Crear directorios
    print("\n📁 Creando directorios necesarios...")
    create_directories()
    
    # Paso 2: Instalar dependencias
    print("\n📦 Instalando dependencias de Python...")
    if not install_requirements():
        success = False
    
    # Paso 3: Descargar modelo YOLO
    print("\n🤖 Configurando modelos de IA...")
    if not download_yolo_model():
        success = False
    
    # Paso 4: Verificar modelo de placas
    if not check_license_plate_model():
        success = False
    
    # Paso 5: Verificar instalación
    if success:
        if verify_installation():
            print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
            print("\n📋 Próximos pasos:")
            print("   1. Asegúrate de tener el archivo 'license_plate_detector.pt'")
            print("   2. Ejecuta 'python run_all.py' para procesar un video")
            print("   3. O ejecuta 'python main.py' para solo detección")
        else:
            print("\n⚠️ Instalación completada con advertencias")
            success = False
    
    if not success:
        print("\n❌ Instalación incompleta. Revisa los errores anteriores.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
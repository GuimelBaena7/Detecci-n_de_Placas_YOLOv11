#!/usr/bin/env python3
"""
Script principal para ejecutar todo el pipeline de detección de placas
"""
import os
import subprocess
import sys

def run_script(script_name, description):
    """Ejecutar un script y manejar errores"""
    print(f"\n🚀 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {description} completado exitosamente")
            return True
        else:
            print(f"❌ Error en {description}")
            return False
            
    except Exception as e:
        print(f"⚠️ Error ejecutando {script_name}: {e}")
        return False

def main():
    print("🎯 PIPELINE DE DETECCIÓN DE PLACAS YOLOv11")
    print("=" * 60)
    
    # Verificar archivos necesarios
    required_files = ['main.py', 'util.py', 'add_missing_data.py', 'visualize.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return
    
    # Paso 1: Detección principal
    if not run_script('main.py', 'Ejecutando detección de vehículos y placas'):
        print("❌ Falló la detección principal. Deteniendo pipeline.")
        return
    
    # Verificar que se generó el CSV
    if not os.path.exists('test.csv'):
        print("❌ No se generó el archivo test.csv")
        return
    
    # Paso 2: Interpolación de datos
    if run_script('add_missing_data.py', 'Interpolando datos faltantes'):
        print("✅ Interpolación completada")
    else:
        print("⚠️ Error en interpolación, continuando con datos originales")
    
    # Paso 3: Visualización
    if run_script('visualize.py', 'Generando video con visualizaciones'):
        print("✅ Visualización completada")
    else:
        print("⚠️ Error en visualización")
    
    print("\n🎉 PIPELINE COMPLETADO")
    print("📁 Archivos generados:")
    
    files_to_check = ['test.csv', 'test_interpolated.csv', 'out.mp4']
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} (no generado)")

if __name__ == "__main__":
    main()
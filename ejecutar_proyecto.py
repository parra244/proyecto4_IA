# Script ejecutor para Proyecto 4 - Employee Attrition
# Este script ejecuta todo el pipeline de análisis y preparación de datos

import subprocess
import sys
import os

def run_script(script_path, description):
    """Ejecutar un script de Python"""
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Ejecutado exitosamente")
            if result.stdout:
                print("Salida:")
                print(result.stdout)
        else:
            print("❌ Error en la ejecución")
            print("Error:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error al ejecutar {script_path}: {e}")

def main():
    """Función principal que ejecuta todo el pipeline"""
    print("INICIANDO PIPELINE COMPLETO DEL PROYECTO 4")
    print("="*60)
    print("Este script ejecutará:")
    print("1. Análisis Exploratorio de Datos (EDA)")
    print("2. Preparación de Datos")
    print("3. Generación de archivos CSV para ML")
    print("="*60)
    
    # Verificar que los archivos existen
    scripts = [
        ("EDA/eda_completo.py", "Análisis Exploratorio de Datos"),
        ("preparacion_datos/preparacion_datos.py", "Preparación de Datos")
    ]
    
    for script_path, description in scripts:
        if os.path.exists(script_path):
            run_script(script_path, description)
        else:
            print(f"❌ No se encontró el archivo: {script_path}")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("📁 EDA/")
    print("  - eda_completo.py")
    print("  - EDA_Employee_Attrition.ipynb")
    print("  - Gráficos de análisis")
    print("\n📁 preparacion_datos/")
    print("  - preparacion_datos.py")
    print("  - Gráficos de PCA y análisis")
    print("\n📁 Data/csv/")
    print("  - TrainX.csv, TrainY.csv")
    print("  - ValidationX.csv, ValidationY.csv")
    print("  - TestX.csv, TestY.csv")
    print("\n📁 Asistincly/")
    print("  - Archivos de asistencia por fecha")
    print("\n✅ Proyecto listo para entrega")

if __name__ == "__main__":
    main()

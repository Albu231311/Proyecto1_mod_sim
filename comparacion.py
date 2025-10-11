import pandas as pd
import os

def cargar_resultados():
    """Carga los archivos CSV de resultados de GA y LP"""
    archivos_faltantes = []
    
    if not os.path.exists('resultados_GA.csv'):
        archivos_faltantes.append('resultados_GA.csv')
    if not os.path.exists('resultados_LP.csv'):
        archivos_faltantes.append('resultados_LP.csv')
    
    if archivos_faltantes:
        print("ERROR: No se encontraron los siguientes archivos:")
        for archivo in archivos_faltantes:
            print(f"  - {archivo}")
        print("\nPor favor, ejecute primero los algoritmos GA y LP para generar los resultados.")
        return None, None
    
    df_ga = pd.read_csv('resultados_GA.csv')
    df_lp = pd.read_csv('resultados_LP.csv')
    
    return df_ga, df_lp

def seleccionar_archivo():
    """Permite al usuario seleccionar qué archivo CSV comparar"""
    df_ga, df_lp = cargar_resultados()
    
    if df_ga is None or df_lp is None:
        return None, None, None
    
    # Obtener archivos únicos presentes en ambos CSV
    archivos_ga = set(df_ga['Archivo evaluado'].unique())
    archivos_lp = set(df_lp['Archivo evaluado'].unique())
    archivos_comunes = archivos_ga.intersection(archivos_lp)
    
    if not archivos_comunes:
        print("\nERROR: No hay archivos en común entre GA y LP.")
        print("\nArchivos en GA:", list(archivos_ga))
        print("Archivos en LP:", list(archivos_lp))
        print("\nAsegúrese de ejecutar ambos algoritmos con el mismo archivo CSV.")
        return None, None, None
    
    print("\n" + "="*60)
    print("COMPARADOR DE RESULTADOS: GA vs LP")
    print("="*60)
    print("\nArchivos disponibles para comparación:")
    archivos_lista = sorted(list(archivos_comunes))
    for i, archivo in enumerate(archivos_lista, 1):
        print(f"{i}. {archivo}")
    print("="*60)
    
    while True:
        try:
            opcion = int(input(f"\nSeleccione un archivo (1-{len(archivos_lista)}): "))
            if 1 <= opcion <= len(archivos_lista):
                archivo_seleccionado = archivos_lista[opcion - 1]
                return df_ga, df_lp, archivo_seleccionado
            else:
                print(f"Por favor, ingrese un número entre 1 y {len(archivos_lista)}")
        except ValueError:
            print("Por favor, ingrese un número válido")

def comparar_resultados(df_ga, df_lp, archivo_seleccionado):
    """Compara los resultados de GA y LP para un archivo específico"""
    
    # Filtrar datos del archivo seleccionado
    ga_datos = df_ga[df_ga['Archivo evaluado'] == archivo_seleccionado].copy()
    lp_datos = df_lp[df_lp['Archivo evaluado'] == archivo_seleccionado].copy()
    
    if ga_datos.empty:
        print(f"\nNo hay datos de GA para {archivo_seleccionado}")
        return
    
    if lp_datos.empty:
        print(f"\nNo hay datos de LP para {archivo_seleccionado}")
        return
    
    # Convertir columnas a números
    ga_datos['Solución encontrada'] = pd.to_numeric(ga_datos['Solución encontrada'])
    ga_datos['Tiempo de ejecución (segundos)'] = pd.to_numeric(ga_datos['Tiempo de ejecución (segundos)'])
    lp_datos['Solución óptima teórica'] = pd.to_numeric(lp_datos['Solución óptima teórica'])
    lp_datos['Tiempo de ejecución (segundos)'] = pd.to_numeric(lp_datos['Tiempo de ejecución (segundos)'])
    
    # Obtener la solución óptima de LP (usar la mejor si hay múltiples)
    solucion_optima_lp = lp_datos['Solución óptima teórica'].min()
    tiempo_lp = lp_datos['Tiempo de ejecución (segundos)'].iloc[0]
    
    # Obtener las 3 mejores soluciones de GA
    ga_ordenado = ga_datos.nsmallest(3, 'Solución encontrada')
    
    # Crear tabla de comparación
    print("\n" + "="*100)
    print(f"COMPARACIÓN DE RESULTADOS PARA: {archivo_seleccionado}")
    print("="*100)
    
    # Preparar datos para la tabla
    resultados = []
    
    # Agregar solución LP
    resultados.append({
        'Tipo': 'LP',
        'Tiempo (s)': f"{tiempo_lp:.4f}",
        'Sol. Óptima Teórica': f"{solucion_optima_lp:.2f}",
        'Sol. Subóptima Encontrada': f"{solucion_optima_lp:.2f}",
        'Error (%)': "0.00"
    })
    
    # Agregar las 3 mejores soluciones GA
    for i, (idx, row) in enumerate(ga_ordenado.iterrows(), 1):
        sol_ga = row['Solución encontrada']
        tiempo_ga = row['Tiempo de ejecución (segundos)']
        error = ((sol_ga - solucion_optima_lp) / solucion_optima_lp) * 100
        
        resultados.append({
            'Tipo': f'GA-{i}',
            'Tiempo (s)': f"{tiempo_ga:.4f}",
            'Sol. Óptima Teórica': f"{solucion_optima_lp:.2f}",
            'Sol. Subóptima Encontrada': f"{sol_ga:.2f}",
            'Error (%)': f"{error:.2f}"
        })
    
    # Crear DataFrame para mostrar
    df_comparacion = pd.DataFrame(resultados)
    
    # Mostrar tabla formateada
    print(f"\n{'Tipo':<8} | {'Tiempo (s)':>12} | {'Sol. Óptima':>15} | {'Sol. Encontrada':>17} | {'Error (%)':>10}")
    print("-" * 100)
    
    for _, row in df_comparacion.iterrows():
        print(f"{row['Tipo']:<8} | {row['Tiempo (s)']:>12} | {row['Sol. Óptima Teórica']:>15} | "
              f"{row['Sol. Subóptima Encontrada']:>17} | {row['Error (%)']:>10}")
    
    print("="*100)
    
    # Guardar comparación en CSV
    guardar_comparacion(df_comparacion, archivo_seleccionado)

def guardar_comparacion(df_comparacion, archivo_seleccionado):
    """Guarda la tabla de comparación en un archivo CSV"""
    nombre_base = archivo_seleccionado.replace('.csv', '')
    nombre_salida = f'comparacion_{nombre_base}.csv'
    df_comparacion.to_csv(nombre_salida, index=False)
    print(f"\n✓ Tabla de comparación guardada en: {nombre_salida}")

def mostrar_resumen_general():
    """Muestra un resumen general de todos los resultados disponibles"""
    df_ga, df_lp = cargar_resultados()
    
    if df_ga is None or df_lp is None:
        return
    
    print("\n" + "="*60)
    print("RESUMEN GENERAL DE RESULTADOS DISPONIBLES")
    print("="*60)
    
    print("\n--- Resultados GA ---")
    print(f"Total de ejecuciones: {len(df_ga)}")
    if len(df_ga) > 0:
        print("\nArchivos evaluados:")
        for archivo in df_ga['Archivo evaluado'].unique():
            count = len(df_ga[df_ga['Archivo evaluado'] == archivo])
            print(f"  - {archivo}: {count} ejecuciones")
    
    print("\n--- Resultados LP ---")
    print(f"Total de ejecuciones: {len(df_lp)}")
    if len(df_lp) > 0:
        print("\nArchivos evaluados:")
        for archivo in df_lp['Archivo evaluado'].unique():
            count = len(df_lp[df_lp['Archivo evaluado'] == archivo])
            print(f"  - {archivo}: {count} ejecuciones")
    
    print("="*60)

# Programa principal
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPARADOR DE RESULTADOS TSP: GA vs LP")
    print("="*60)
    print("\nEste programa compara los resultados de:")
    print("  - Algoritmo Genético (GA)")
    print("  - Programación Lineal/Optimización (LP)")
    print("="*60)
    
    # Mostrar resumen general
    mostrar_resumen_general()
    
    # Seleccionar archivo y comparar
    df_ga, df_lp, archivo_seleccionado = seleccionar_archivo()
    
    if archivo_seleccionado:
        comparar_resultados(df_ga, df_lp, archivo_seleccionado)
        
        # Preguntar si desea hacer otra comparación
        while True:
            print("\n" + "="*60)
            respuesta = input("¿Desea comparar otro archivo? (s/n): ").strip().lower()
            if respuesta == 's':
                df_ga, df_lp, archivo_seleccionado = seleccionar_archivo()
                if archivo_seleccionado:
                    comparar_resultados(df_ga, df_lp, archivo_seleccionado)
                else:
                    break
            else:
                print("\n¡Gracias por usar el comparador!")
                break
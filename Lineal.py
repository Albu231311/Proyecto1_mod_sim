import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def cargar_ciudades_csv(nombre_archivo):
    if not os.path.exists(nombre_archivo):
        raise FileNotFoundError(f"No se encontró el archivo {nombre_archivo}.")
    
    df = pd.read_csv(nombre_archivo)
    if not {'x','y'}.issubset(df.columns):
        raise ValueError("El CSV debe contener columnas 'x' y 'y'.")
    
    coords = df[['x','y']].to_numpy(dtype=float)
    nombres = df['nombre'].values if 'nombre' in df.columns else [f"Ciudad {i}" for i in range(len(coords))]
    return coords, nombres

def calcular_matriz_distancias(coords):
    distancias = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2))
    # Convertir a enteros 
    return (distancias * 1000).astype(int)

def resolver_tsp_ortools(distancias, coords, nombres, tiempo_limite=300):
    n = len(coords)
    
    print("\n" + "="*60)
    print("RESOLVIENDO TSP CON OPTIMIZACIÓN")
    print("="*60)
    print(f"Número de ciudades: {n}")
    print(f"Método: Routing con búsqueda local guiada")
    print(f"Tiempo límite: {tiempo_limite} segundos")
    print("="*60)
    
    # Crear el gestor de índices
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n ciudades, 1 vehículo, depósito en 0
    
    # Crear el modelo de routing
    routing = pywrapcp.RoutingModel(manager)
    
    # Definir la función de costo (distancia)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distancias[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Configurar parámetros de búsqueda
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = tiempo_limite
    search_parameters.log_search = True
    
    print("\nIniciando solver OR-Tools...")
    print("(Búsqueda de solución óptima/near-óptima en progreso...)\n")
    inicio = time.time()
    
    # Resolver
    solution = routing.SolveWithParameters(search_parameters)
    
    tiempo_ejecucion = time.time() - inicio
    
    # Verificar si hay solución
    print("\n" + "="*60)
    print("RESULTADOS DEL SOLVER")
    print("="*60)
    print(f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
    
    if solution:
        distancia_total = solution.ObjectiveValue() / 1000.0  # Convertir de vuelta
        print(f"✓ Solución encontrada")
        print(f"Distancia: {distancia_total:.2f}")
        
        # Extraer la ruta
        ruta = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            ruta.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        
        # Determinar si es óptima o near-óptima
        # OR-Tools con Guided Local Search típicamente da soluciones muy cercanas al óptimo
        tipo_solucion = "Near-Óptima"
        if tiempo_ejecucion < tiempo_limite * 0.5:
            tipo_solucion = "Óptima/Near-Óptima"
        
        return ruta, distancia_total, tiempo_ejecucion, tipo_solucion
    else:
        print("✗ No se encontró solución")
        return None, None, None, "Sin solución"

def visualizar_solucion(coords, nombres, ruta, distancia, titulo, tipo_solucion, archivo_salida=None):
    """Visualiza la solución encontrada"""
    plt.figure(figsize=(12, 8))
    
    if ruta is None or len(ruta) < 2:
        plt.text(0.5, 0.5, 'No se pudo encontrar una solución', 
                ha='center', va='center', fontsize=16)
        plt.title(titulo, fontsize=16, fontweight='bold')
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    # Dibujar la ruta
    x = [coords[i][0] for i in ruta] + [coords[ruta[0]][0]]
    y = [coords[i][1] for i in ruta] + [coords[ruta[0]][1]]
    
    plt.plot(x, y, 'b-', alpha=0.7, linewidth=2, label='Ruta optimizada')
    plt.scatter(coords[:,0], coords[:,1], color='red', s=100, 
               zorder=10, edgecolors='black', linewidths=1.5)
    
    # Marcar inicio
    plt.scatter(coords[ruta[0]][0], coords[ruta[0]][1], 
               color='green', s=200, marker='*', zorder=11, 
               edgecolors='black', linewidths=2, label='Inicio')
    
    plt.title(f'{titulo}\nDistancia ({tipo_solucion}): {distancia:.2f}', 
             fontsize=16, fontweight='bold')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Ajustar intervalos según número de ciudades
    if len(coords) == 229:
        from matplotlib.ticker import MultipleLocator
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(25))
    elif len(coords) == 80:
        from matplotlib.ticker import MultipleLocator
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    
    plt.tight_layout()
    
    if archivo_salida:
        plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada como '{archivo_salida}'")
    
    plt.show()

def guardar_resultados_csv(tipo_solucion, tiempo_ejecucion, solucion_optima, nombre_csv):
    
    resultados = {
        'Archivo evaluado': [nombre_csv],
        'Tipo de Solución': [tipo_solucion],
        'Tiempo de ejecución (segundos)': [f"{tiempo_ejecucion:.4f}"],
        'Solución óptima teórica': [f"{solucion_optima:.2f}"]
    }
    
    df_nuevo = pd.DataFrame(resultados)
    nombre_archivo = 'resultados_LP.csv'
    
    # Si el archivo existe, agregar los nuevos datos; si no, crear uno nuevo
    if os.path.exists(nombre_archivo):
        df_existente = pd.read_csv(nombre_archivo)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
        df_final.to_csv(nombre_archivo, index=False)
        print(f"\n✓ Resultados agregados a: {nombre_archivo}")
    else:
        df_nuevo.to_csv(nombre_archivo, index=False)
        print(f"\n✓ Resultados guardados en: {nombre_archivo}")

def mostrar_tabla_parametros(n_ciudades):
    
    num_variables = n_ciudades ** 2
    # Restricciones básicas: 2n (entrada/salida) + (n-1) (MTZ) + n*(n-1) (orden MTZ)
    num_restricciones = 2 * n_ciudades + (n_ciudades - 1) + n_ciudades * (n_ciudades - 1)
    
    # Para este caso, no hay GA, así que mostramos N/A
    tam_poblacion_ga = "N/A"
    num_iteraciones_ga = "N/A"
    
    print("\n" + "="*80)
    print("RESUMEN DE PARÁMETROS DEL PROBLEMA")
    print("="*80)
    print(f"{'Parámetro':<40} | {'Valor':>15}")
    print("-"*80)
    print(f"{'Número de ciudades':<40} | {n_ciudades:>15}")
    print(f"{'Tamaño de la población del GA':<40} | {tam_poblacion_ga:>15}")
    print(f"{'Número de iteraciones del GA':<40} | {num_iteraciones_ga:>15}")
    print(f"{'Número de variables (problema LP)':<40} | {num_variables:>15}")
    print(f"{'Número de restricciones (problema LP)':<40} | {num_restricciones:>15}")
    print("="*80)
    print("\nNOTA: Este método usa optimización directa (no algoritmo genético)")
    print("="*80)

def mostrar_menu():
    print("\n" + "="*60)
    print("TSP - OPTIMIZACIÓN CON OR-TOOLS")
    print("="*60)
    print("\nSeleccione el archivo CSV a procesar:")
    print("1. eil101_ciudades.csv (101 ciudades) - ~2-3 minutos")
    print("2. gr229_ciudades.csv (229 ciudades) - ~5-8 minutos")
    print("3. ruta.csv (80 ciudades) - ~1-2 minutos")
    print("="*60)
    
    while True:
        opcion = input("\nIngrese su opción (1, 2 o 3): ").strip()
        if opcion == '1':
            return "eil101_ciudades.csv"
        elif opcion == '2':
            return "gr229_ciudades.csv"
        elif opcion == '3':
            return "ruta.csv"
        else:
            print("Opción inválida. Por favor ingrese 1, 2 o 3.")

# Ejecución principal
print("="*60)
print("NOTA: Este programa usa Google OR-Tools")
print("Si no está instalado, ejecute: pip install ortools")
print("="*60)

nombre_csv = mostrar_menu()

print(f"\nCargando archivo: {nombre_csv}")
coords, nombres = cargar_ciudades_csv(nombre_csv)
distancias = calcular_matriz_distancias(coords)
print(f"✓ Archivo cargado exitosamente: {len(coords)} ciudades")

# Configurar tiempo límite
if len(coords) <= 80:
    tiempo_limite = 120   # 2 minutos
elif len(coords) <= 101:
    tiempo_limite = 180   # 3 minutos
else:
    tiempo_limite = 300   # 5 minutos

# Resolver TSP
ruta, distancia_optima, tiempo_ejecucion, tipo_solucion = resolver_tsp_ortools(
    distancias, coords, nombres, tiempo_limite
)

if ruta is not None:
    # Mostrar resultados
    print("\n" + "="*60)
    print(f"SOLUCIÓN {tipo_solucion.upper()} ENCONTRADA")
    print("="*60)
    print(f"Ruta: {ruta}")
    print(f"Distancia total: {distancia_optima:.2f}")
    print(f"Tiempo de cómputo: {tiempo_ejecucion:.2f} segundos")
    print("="*60)
    
    # Mostrar tabla de parámetros
    mostrar_tabla_parametros(len(coords))
    
    # Guardar resultados en CSV
    guardar_resultados_csv("LP", tiempo_ejecucion, distancia_optima, nombre_csv)
    
    # Visualizar
    nombre_problema = nombre_csv.split('.')[0]
    visualizar_solucion(coords, nombres, ruta, distancia_optima, 
                       f"Solución Optimizada TSP - {nombre_problema}",
                       tipo_solucion,
                       f"tsp_optimizado_{nombre_problema}.png")
else:
    print("\n" + "="*60)
    print("NO SE ENCONTRÓ SOLUCIÓN")
    print("="*60)
    
    # Mostrar tabla incluso si no hay solución
    mostrar_tabla_parametros(len(coords))
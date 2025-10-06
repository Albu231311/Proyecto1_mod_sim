import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

def cargar_ciudades_csv(nombre_archivo):
    if not os.path.exists(nombre_archivo):
        raise FileNotFoundError(f"No se encontró el archivo {nombre_archivo}.")
    
    df = pd.read_csv(nombre_archivo)
    if not {'x','y'}.issubset(df.columns):
        raise ValueError("El CSV debe contener columnas 'x' y 'y'.")
    
    coords = df[['x','y']].to_numpy(dtype=float)
    return coords

def calcular_matriz_distancias(coords):
    return np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2))

def distancia_total(ruta, distancias):
    return sum(distancias[ruta[i], ruta[(i+1)%len(ruta)]] for i in range(len(ruta)))

def crear_poblacion(N, n_ciudades):
    return [random.sample(range(n_ciudades), n_ciudades) for _ in range(N)]

def seleccion(poblacion, fitness, n_seleccionados):
    idx = np.argsort(fitness)[:n_seleccionados]
    return [poblacion[i] for i in idx]

def cruce_ox(padre1, padre2):
    a, b = sorted(random.sample(range(len(padre1)), 2))
    hijo = [None]*len(padre1)
    hijo[a:b] = padre1[a:b]
    ptr = b
    for ciudad in padre2:
        if ciudad not in hijo:
            if ptr == len(hijo): ptr = 0
            hijo[ptr] = ciudad
            ptr += 1
    return hijo

def mutacion_swap(ruta):
    nueva_ruta = ruta.copy()
    a, b = random.sample(range(len(nueva_ruta)), 2)
    nueva_ruta[a], nueva_ruta[b] = nueva_ruta[b], nueva_ruta[a]
    return nueva_ruta

def mutacion_inversion(ruta):
    nueva_ruta = ruta.copy()
    a, b = sorted(random.sample(range(len(nueva_ruta)), 2))
    nueva_ruta[a:b+1] = reversed(nueva_ruta[a:b+1])
    return nueva_ruta

def inicializar_visualizacion(coords, num_ciudades):
    plt.ion()
    
    if num_ciudades == 229:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle('Algoritmo Genético - TSP (Evolución en Tiempo Real)', 
                     fontsize=16, fontweight='bold')
        return fig, ax1, None, None
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Algoritmo Genético - TSP (Evolución en Tiempo Real)', 
                     fontsize=16, fontweight='bold')
        return fig, ax1, ax2, ax3

def actualizar_visualizacion(ax1, ax2, ax3, ruta, coords, distancias, 
                            iter, mejor_dist, historial, mejor_historico):
    ax1.clear()
    if ax2 is not None:
        ax2.clear()
    if ax3 is not None:
        ax3.clear()
    
    if ruta:
        x = [coords[c][0] for c in ruta] + [coords[ruta[0]][0]]
        y = [coords[c][1] for c in ruta] + [coords[ruta[0]][1]]
        
        ax1.plot(x, y, 'b-', alpha=0.7, linewidth=1.5, zorder=1)
        ax1.scatter(coords[:,0], coords[:,1], color='red', s=80, 
                   zorder=10, edgecolors='black', linewidths=1)
    else:
        ax1.scatter(coords[:,0], coords[:,1], color='red', s=80, 
                   zorder=10, edgecolors='black', linewidths=1)
    
    ax1.set_title(f'Ruta en Evolución (Distancia: {mejor_dist:.2f})', 
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    if len(coords) == 229:
        from matplotlib.ticker import MultipleLocator
        ax1.xaxis.set_major_locator(MultipleLocator(10))
        ax1.yaxis.set_major_locator(MultipleLocator(25))
        
        min_y = coords[:, 1].min()
        max_y = coords[:, 1].max()
        margen_y = (max_y - min_y) * 0.05
        ax1.set_ylim(min_y - margen_y, max_y + margen_y)
    elif len(coords) == 80:
        from matplotlib.ticker import MultipleLocator
        ax1.xaxis.set_major_locator(MultipleLocator(10))
        ax1.yaxis.set_major_locator(MultipleLocator(10))
        ax1.set_aspect('equal', adjustable='box')
    else:
        ax1.set_aspect('equal', adjustable='box')
    
    if ax2 is not None and historial:
        ax2.plot(historial, 'b-', linewidth=2, label='Mejor actual')
        ax2.axhline(y=mejor_historico, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Mejor global: {mejor_historico:.2f}')
        ax2.set_title('Convergencia del Algoritmo', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Distancia')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    if ax3 is not None and len(historial) > 1:
        mejora = [(historial[0] - d) / historial[0] * 100 for d in historial]
        ax3.plot(mejora, 'g-', linewidth=2)
        ax3.set_title('Mejora Porcentual', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('Mejora (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.pause(0.001)

def algoritmo_genetico_tsp(distancias, coords, N, maxIter, pct_seleccion, pct_cruce, pct_mutacion):
    n_ciudades = distancias.shape[0]
    poblacion = crear_poblacion(N, n_ciudades)
    mejor_ruta = None
    mejor_dist = float('inf')
    historial = []
    
    fig, ax1, ax2, ax3 = inicializar_visualizacion(coords, n_ciudades)
    
    print("INICIANDO ALGORITMO GENÉTICO")
    print("="*60)

    for iter in range(maxIter):
        fitness = [distancia_total(r, distancias) for r in poblacion]

        idx_best = np.argmin(fitness)
        if fitness[idx_best] < mejor_dist:
            mejor_dist = fitness[idx_best]
            mejor_ruta = poblacion[idx_best].copy()

        historial.append(mejor_dist)
        
        if (iter + 1) % 50 == 0 or iter == 0:
            mejora_pct = ((historial[0] - mejor_dist) / historial[0] * 100) if historial else 0
            print(f"Iteración {iter+1}/{maxIter} | Mejor distancia: {mejor_dist:.2f} | Mejora: {mejora_pct:.1f}%")
        
        if (iter % 5 == 0 or iter == maxIter - 1):
            actualizar_visualizacion(ax1, ax2, ax3, mejor_ruta, coords, 
                                   distancias, iter+1, mejor_dist, historial, 
                                   min(historial))

        n_sel = max(2, int(N * pct_seleccion))
        seleccionados = seleccion(poblacion, fitness, n_sel)

        n_cruce = int(N * pct_cruce)
        hijos = []
        while len(hijos) < n_cruce:
            p1, p2 = random.sample(seleccionados, 2)
            hijos.append(cruce_ox(p1, p2))

        n_mut = int(N * pct_mutacion)
        mutados = []
        for i in range(n_mut):
            ruta = random.choice(seleccionados)
            if i % 2 == 0:
                mutados.append(mutacion_swap(ruta))
            else:
                mutados.append(mutacion_inversion(ruta))

        poblacion = [s.copy() for s in seleccionados] + hijos + mutados
        while len(poblacion) < N:
            poblacion.append(random.sample(range(n_ciudades), n_ciudades))

    actualizar_visualizacion(ax1, ax2, ax3, mejor_ruta, coords, 
                           distancias, maxIter, mejor_dist, historial, 
                           min(historial))
    plt.ioff()
    plt.savefig('simulacion_final.png', dpi=300, bbox_inches='tight')
    print("\nSimulación guardada como 'simulacion_final.png'")
    print("Algoritmo completado")
    
    return mejor_ruta, mejor_dist, historial

def mostrar_menu():
    print("ALGORITMO GENÉTICO PARA TSP")
    print("="*60)
    print("\nSeleccione el archivo CSV a procesar:")
    print("1. eil101_ciudades.csv (101 ciudades)")
    print("2. gr229_ciudades.csv (229 ciudades)")
    print("3. ruta.csv (80 ciudades)")
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
nombre_csv = mostrar_menu()

print(f"\nCargando archivo: {nombre_csv}")
coords = cargar_ciudades_csv(nombre_csv)
distancias = calcular_matriz_distancias(coords)
print(f"Archivo cargado exitosamente: {len(coords)} ciudades")

print("\n" + "="*60)
print("CONFIGURACIÓN DE PARÁMETROS")
print("="*60)
N = int(input("Tamaño de la población (N): "))
maxIter = int(input("Número máximo de iteraciones: "))
pct_seleccion = float(input("Porcentaje de sobrevivientes (0-1, ej: 0.2): "))
pct_cruce = float(input("Porcentaje de cruce (0-1, ej: 0.6): "))
pct_mutacion = float(input("Porcentaje de mutación (0-1, ej: 0.2): "))

best, D, historial = algoritmo_genetico_tsp(distancias, coords, N, maxIter, 
                                            pct_seleccion, pct_cruce, pct_mutacion)

print("\nRESULTADOS FINALES")
print("="*60)
print("Mejor ruta encontrada:", best)
print("Distancia total: {:.2f}".format(D))
print("="*60)
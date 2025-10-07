import os
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import nsmallest

ESCENARIOS_DIR = 'escenarios'
RESULTADOS_DIR = 'resultados'

def asegurar_directorio(path):
    os.makedirs(path, exist_ok=True)

def cargar_ciudades_csv(nombre_archivo):
    if os.path.isabs(nombre_archivo) or os.path.dirname(nombre_archivo):
        ruta = nombre_archivo
    else:
        ruta = os.path.join(ESCENARIOS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo {ruta}.")
    df = pd.read_csv(ruta)
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
    nueva_ruta[a:b+1] = list(reversed(nueva_ruta[a:b+1]))
    return nueva_ruta

def twoOptSwap(route, i, k):
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def twoOpt(route, distancias, max_iter=200):
    """Búsqueda local 2-opt limitada por max_iter (rápida)."""
    if route is None or len(route) < 4:
        return route, distancia_total(route, distancias) if route else (route, None)
    best = route.copy()
    best_dist = distancia_total(best, distancias)
    n = len(best)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                new_route = twoOptSwap(best, i, k)
                new_dist = distancia_total(new_route, distancias)
                if new_dist + 1e-12 < best_dist:
                    best = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best, best_dist

# ----------------detección y conteo de cruces
def _segment_intersect(p1, p2, p3, p4):
    import numpy as _np

    try:
        t1 = (float(p1[0]), float(p1[1]))
        t2 = (float(p2[0]), float(p2[1]))
        t3 = (float(p3[0]), float(p3[1]))
        t4 = (float(p4[0]), float(p4[1]))
    except Exception:
        if _np.array_equal(p1, p3) or _np.array_equal(p1, p4) or _np.array_equal(p2, p3) or _np.array_equal(p2, p4):
            return False
        t1, t2, t3, t4 = tuple(p1), tuple(p2), tuple(p3), tuple(p4)

    if t1 == t3 or t1 == t4 or t2 == t3 or t2 == t4:
        return False

    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    o1 = orient(t1, t2, t3)
    o2 = orient(t1, t2, t4)
    o3 = orient(t3, t4, t1)
    o4 = orient(t3, t4, t2)

    return (o1 * o2 < 0) and (o3 * o4 < 0)


def contar_cruces(ruta, coords):
    import numpy as _np

    n = len(ruta)
    if n < 4:
        return 0
    count = 0
    coords_arr = _np.asarray(coords)
    for i in range(n):
        a1 = coords_arr[ruta[i]]
        a2 = coords_arr[ruta[(i+1) % n]]
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue
            b1 = coords_arr[ruta[j]]
            b2 = coords_arr[ruta[(j+1) % n]]
            try:
                if _segment_intersect(a1, a2, b1, b2):
                    count += 1
            except Exception:
                continue
    return count

def save_route_plot(coords, ruta, filename, title=None, equal_aspect=True):
    fig, ax = plt.subplots(figsize=(8,8))
    if ruta:
        x = [coords[c][0] for c in ruta] + [coords[ruta[0]][0]]
        y = [coords[c][1] for c in ruta] + [coords[ruta[0]][1]]
        ax.plot(x, y, '-', linewidth=1.2, zorder=1)
    ax.scatter(coords[:,0], coords[:,1], c='red', s=30, zorder=2, edgecolors='black', linewidths=0.4)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

def inicializar_visualizacion(coords, num_ciudades):
    plt.ion()
    if num_ciudades >= 200:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Algoritmo Genético - TSP (Evolución en Tiempo Real)', fontsize=16, fontweight='bold')
    return fig, ax1, ax2, ax3

def actualizar_visualizacion(ax1, ax2, ax3, ruta, coords, distancias, iter, mejor_dist, historial, mejor_historico):
    ax1.clear(); ax2.clear(); ax3.clear()
    if ruta:
        x = [coords[c][0] for c in ruta] + [coords[ruta[0]][0]]
        y = [coords[c][1] for c in ruta] + [coords[ruta[0]][1]]
        ax1.plot(x, y, alpha=0.9, linewidth=1.2, zorder=1)
        ax1.scatter(coords[:,0], coords[:,1], s=40, zorder=10, edgecolors='black', linewidths=0.5)
    else:
        ax1.scatter(coords[:,0], coords[:,1], s=40, zorder=10, edgecolors='black', linewidths=0.5)
    ax1.set_title(f'Ruta en Evolución (Distancia: {mejor_dist:.2f})', fontweight='bold', fontsize=12)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.grid(True, alpha=0.3)
    if ax2 is not None and historial:
        ax2.plot(historial, linewidth=2, label='Mejor actual')
        ax2.axhline(y=mejor_historico, color='r', linestyle='--', linewidth=1.2, label=f'Mejor global: {mejor_historico:.2f}')
        ax2.set_title('Convergencia del Algoritmo'); ax2.set_xlabel('Iteración'); ax2.set_ylabel('Distancia'); ax2.grid(True, alpha=0.3); ax2.legend()
    if ax3 is not None and len(historial) > 1:
        mejora = [(historial[0] - d) / historial[0] * 100 for d in historial]
        ax3.plot(mejora, linewidth=2); ax3.set_title('Mejora Porcentual'); ax3.set_xlabel('Iteración'); ax3.set_ylabel('Mejora (%)'); ax3.grid(True, alpha=0.3); ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout(); plt.pause(0.001)

# ----------------algoritmo genético
def algoritmo_genetico_tsp(distancias, coords, N, maxIter, pct_seleccion, pct_cruce, pct_mutacion,
                           collectTopK=False, topK=3, seed=None, aplicar2Opt=True, output_dir=None, rep_id=1):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    n_ciudades = distancias.shape[0]
    poblacion = crear_poblacion(N, n_ciudades)
    mejor_ruta = None; mejor_dist = float('inf'); historial = []
    mejores_vistos = {}
    fig, ax1, ax2, ax3 = inicializar_visualizacion(coords, n_ciudades)
    max_iter_local = 4
    probAplicar2Opt = 0.045
    print("INIT ALGORITMO GENÉTICO")
    print("="*60)
    for it in range(maxIter):
        fitness = [distancia_total(r, distancias) for r in poblacion]
        idx_best = np.argmin(fitness)
        if fitness[idx_best] < mejor_dist:
            mejor_dist = fitness[idx_best]; mejor_ruta = poblacion[idx_best].copy()
        ruta_tuple = tuple(poblacion[idx_best])
        mejores_vistos[ruta_tuple] = min(mejores_vistos.get(ruta_tuple, float('inf')), fitness[idx_best])
        historial.append(mejor_dist)
        if (it + 1) % 50 == 0 or it == 0:
            mejora_pct = ((historial[0] - mejor_dist) / historial[0] * 100) if historial else 0
            print(f"Iteración {it+1}/{maxIter} | Mejor distancia: {mejor_dist:.2f} | Mejora: {mejora_pct:.1f}%")
        if (it % 5 == 0 or it == maxIter - 1):
            actualizar_visualizacion(ax1, ax2, ax3, mejor_ruta, coords, distancias, it+1, mejor_dist, historial, min(historial))
        n_sel = max(2, int(N * pct_seleccion))
        seleccionados = seleccion(poblacion, fitness, n_sel)
        n_cruce = max(0, int(N * pct_cruce))
        if n_cruce == 0 and pct_cruce > 0:
            n_cruce = 1
        hijos = []
        while len(hijos) < n_cruce:
            p1, p2 = random.sample(seleccionados, 2)
            hijo = cruce_ox(p1, p2)
            hijos.append(hijo)
        n_mut = max(0, int(N * pct_mutacion))
        if n_mut == 0 and pct_mutacion > 0:
            n_mut = 1
        mutados = []
        for i in range(n_mut):
            ruta = random.choice(seleccionados)
            mutado = mutacion_swap(ruta) if (i % 2 == 0) else mutacion_inversion(ruta)
            mutados.append(mutado)
        hijos_mejorados = []
        for h in hijos:
            if aplicar2Opt and random.random() <= probAplicar2Opt:
                h_opt, _ = twoOpt(h, distancias, max_iter=max_iter_local)
                hijos_mejorados.append(h_opt)
            else:
                hijos_mejorados.append(h)
        mutados_mejorados = []
        for m in mutados:
            if aplicar2Opt and random.random() <= probAplicar2Opt:
                m_opt, _ = twoOpt(m, distancias, max_iter=max_iter_local)
                mutados_mejorados.append(m_opt)
            else:
                mutados_mejorados.append(m)
        hijos = hijos_mejorados; mutados = mutados_mejorados
        poblacion = [s.copy() for s in seleccionados] + hijos + mutados
        while len(poblacion) < N:
            poblacion.append(random.sample(range(n_ciudades), n_ciudades))
    if aplicar2Opt and mejor_ruta is not None:
        mejor_ruta, mejor_dist = twoOpt(mejor_ruta, distancias, max_iter=80)
    plt.ioff()
    if output_dir:
        asegurar_directorio(output_dir)
        nombre_fig = os.path.join(output_dir, f'sim_ga_{n_ciudades}_rep{rep_id}.png')
        try:
            fig.savefig(nombre_fig, dpi=300, bbox_inches='tight')
        except Exception:
            pass
    try:
        plt.close(fig)
    except Exception:
        pass
    print("\nAlgoritmo completado")
    if collectTopK:
        topk = sorted(mejores_vistos.items(), key=lambda x: x[1])[:topK]
        topk_improved = []
        for ruta_tuple, dist in topk:
            ruta_list = list(ruta_tuple)
            if aplicar2Opt:
                ruta_opt, dist_opt = twoOpt(ruta_list, distancias, max_iter=500)
                topk_improved.append((tuple(ruta_opt), dist_opt))
            else:
                topk_improved.append((ruta_tuple, dist))
        return mejor_ruta, mejor_dist, historial, topk_improved
    else:
        return mejor_ruta, mejor_dist, historial

def ejecutarGA(nombre_csv, N, maxIter, pct_seleccion, pct_cruce, pct_mutacion,
               seed=None, aplicar2Opt=True, k_repeticiones=1, verbose=True):
    base = os.path.basename(nombre_csv).replace('.csv','')
    input_path = os.path.join(ESCENARIOS_DIR, os.path.basename(nombre_csv)) if not os.path.isabs(nombre_csv) else nombre_csv
    output_dir = os.path.join(RESULTADOS_DIR, base)
    asegurar_directorio(output_dir)
    coords = cargar_ciudades_csv(input_path)
    distancias = calcular_matriz_distancias(coords)
    resultados = []
    seeds = [seed + i if seed is not None else None for i in range(k_repeticiones)]
    for i in range(k_repeticiones):
        if verbose:
            print(f"\n--- Repetición {i+1}/{k_repeticiones} para {base} ---")
        t0 = time.perf_counter()
        best, D, historial, top3 = algoritmo_genetico_tsp(distancias, coords, N, maxIter, pct_seleccion, pct_cruce, pct_mutacion,
                                                         collectTopK=True, topK=3, seed=seeds[i], aplicar2Opt=aplicar2Opt,
                                                         output_dir=output_dir, rep_id=i+1)
        t1 = time.perf_counter()
        runtime = t1 - t0
        #guardar top3 como CSV
        top3_rows = []
        for rank, (ruta_tuple, dist_val) in enumerate(top3, start=1):
            top3_rows.append({'rank': rank, 'distancia': dist_val, 'ruta': list(ruta_tuple), 'cruces': contar_cruces(list(ruta_tuple), coords)})
        df_top3 = pd.DataFrame(top3_rows)
        df_top3.to_csv(os.path.join(output_dir, f'top3_rep_{i+1}.csv'), index=False)
        nombre_fig_ruta = os.path.join(output_dir, f'ga_mejor_{base}_rep{i+1}.png')
        save_route_plot(coords, best, nombre_fig_ruta, title=f'GA Mejor - {base} rep{i+1}')
        cruces_mejor = contar_cruces(best, coords)
        resultados.append({'mejorRuta': best, 'distancia': D, 'historial': historial, 'top3': top3, 'runtime': runtime, 'seed': seeds[i], 'cruces': cruces_mejor, 'fig_ga': nombre_fig_ruta, 'top3_csv': os.path.join(output_dir, f'top3_rep_{i+1}.csv')})
    #stats
    distancias_list = [r['distancia'] for r in resultados]
    runtime_list = [r['runtime'] for r in resultados]
    estadisticas = {
        'distancia_media': float(np.mean(distancias_list)),
        'distancia_std': float(np.std(distancias_list)),
        'distancia_min': float(np.min(distancias_list)),
        'distancia_max': float(np.max(distancias_list)),
        'runtime_medio': float(np.mean(runtime_list)),
        'runtime_total': float(sum(runtime_list)),
        'n_ciudades': len(coords),
        'resultados': resultados,
        'output_dir': output_dir
    }
    return estadisticas

# ---------------- menú
def listar_escenarios():
    asegurar_directorio(ESCENARIOS_DIR)
    files = [f for f in os.listdir(ESCENARIOS_DIR) if f.endswith('.csv')]
    files.sort()
    return files

def pedir_entero(prompt, minimo=None, maximo=None, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if val == '' and default is not None:
                return default
            iv = int(val)
            if minimo is not None and iv < minimo:
                print("Valor muy pequeño.")
                continue
            if maximo is not None and iv > maximo:
                print("Valor muy grande.")
                continue
            return iv
        except KeyboardInterrupt:
            raise
        except:
            print("Entrada inválida. Ingrese un entero.")

def pedir_float(prompt, minimo=None, maximo=None, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if val == '' and default is not None:
                return default
            fv = float(val)
            if minimo is not None and fv < minimo:
                print("Valor demasiado pequeño.")
                continue
            if maximo is not None and fv > maximo:
                print("Valor demasiado grande.")
                continue
            return fv
        except KeyboardInterrupt:
            raise
        except:
            print("Entrada inválida. Ingrese un número (ej: 0.2).")

def mostrar_menu():
    while True:
        escenarios = listar_escenarios()
        print("\n" + "="*60)
        print("ALGORITMO GENÉTICO PARA TSP - MENÚ")
        print("="*60)
        if not escenarios:
            print(f"No hay archivos .csv en la carpeta '{ESCENARIOS_DIR}'. Coloca los escenarios ahí.")
            return None
        for idx, f in enumerate(escenarios, start=1):
            print(f"{idx}. {f}")
        print("0. Salir")
        print("="*60)
        opcion = input("Selecciona el número del escenario que deseas ejecutar (0 para salir): ").strip()
        if opcion == '0':
            return None
        try:
            i = int(opcion)
            if 1 <= i <= len(escenarios):
                return escenarios[i-1]
            else:
                print("Opción fuera de rango.")
        except:
            print("Opción inválida. Intenta otra vez.")

if __name__ == "__main__":
    print("TSP - Algoritmo Genético (memetic 2-opt) - Interfaz interactiva")
    while True:
        try:
            escenario = mostrar_menu()
            if escenario is None:
                print("Saliendo.")
                break
            print(f"\nHas seleccionado: {escenario}")
            N = pedir_entero("Tamaño de la población N (ej: 100) [enter para 100]: ", minimo=2, default=100)
            maxIter = pedir_entero("Número máximo de iteraciones (ej: 500) [enter para 500]: ", minimo=1, default=500)
            pct_seleccion = pedir_float("Porcentaje de sobrevivientes (0-1, ej: 0.2) [enter para 0.2]: ", minimo=0.0, maximo=1.0, default=0.2)
            pct_cruce = pedir_float("Porcentaje de cruce (0-1, ej: 0.6) [enter para 0.6]: ", minimo=0.0, maximo=1.0, default=0.6)
            pct_mutacion = pedir_float("Porcentaje de mutación (0-1, ej: 0.2) [enter para 0.2]: ", minimo=0.0, maximo=1.0, default=0.2)
            k = pedir_entero("Número de repeticiones (k) para estadísticas [enter para 1]: ", minimo=1, default=1)
            seed_input = input("Seed (entero) o enter para aleatorio: ").strip()
            seed = int(seed_input) if seed_input != '' else None
            print("\nEjecutando GA... (salidas en carpeta resultados/<escenario>/)")
            resumen = ejecutarGA(escenario, N, maxIter, pct_seleccion, pct_cruce, pct_mutacion, seed=seed, aplicar2Opt=True, k_repeticiones=k, verbose=True)
            print("\nResumen GA:")
            print(f" - Carpeta de resultados: {resumen['output_dir']}")
            print(f" - Distancia media: {resumen['distancia_media']:.2f} (std {resumen['distancia_std']:.2f})")
            continuar = input("\n¿Deseas ejecutar otro escenario? (s/n) [s]: ").strip().lower()
            if continuar == '' or continuar.startswith('s'):
                continue
            else:
                print("Terminando sesión.")
                break
        except KeyboardInterrupt:
            print("\nInterrumpido por usuario. Saliendo.")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            print("Volviendo al menú.")
            continue

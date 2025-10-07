import os
import time
import numpy as np
import pulp
import matplotlib.pyplot as plt
from TSP import cargar_ciudades_csv, calcular_matriz_distancias, asegurar_directorio, RESULTADOS_DIR

def solve_tsp_pulp(distancias, timeLimit=600, msg=False):
    n = distancias.shape[0]
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    x = pulp.LpVariable.dicts('x', (range(n), range(n)), lowBound=0, upBound=1, cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts('u', range(n), lowBound=0, upBound=n-1, cat=pulp.LpInteger)

    prob += pulp.lpSum(distancias[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n) if j != i) == 1
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(n) if i != j) == 1

    prob += u[0] == 0
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n) * x[i][j] <= n - 1

    n_binarias = n * (n - 1)
    n_mtz = n
    n_vars = n_binarias + n_mtz
    n_constraints = 2 * n + (n - 1) * (n - 2) + 1

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=timeLimit)
    t0 = time.perf_counter()
    prob.solve(solver)
    t1 = time.perf_counter()

    status = pulp.LpStatus.get(prob.status, 'Unknown')
    val = pulp.value(prob.objective) if prob.status == 1 else None
    runtime = t1 - t0

    ruta = []
    if prob.status in (1, 0, -1):  # 1=Optimal, 0=Not Solved?, -1=Undefined
        xsol = [[int(pulp.value(x[i][j]) or 0) for j in range(n)] for i in range(n)]
        cur = 0
        ruta = [cur]
        visited = set(ruta)
        for _ in range(n-1):
            nxt = [j for j in range(n) if xsol[cur][j] == 1]
            if not nxt:
                break
            cur = nxt[0]
            ruta.append(cur)
            if cur in visited:
                break
            visited.add(cur)
    return {'ruta': ruta, 'valor': val, 'runtime': runtime, 'status': status, 'n_vars': n_vars, 'n_constraints': n_constraints}

def visualizar_ruta_lp(coords, ruta, nombre_escenario, output_dir):
    asegurar_directorio(output_dir)
    plt.figure(figsize=(8,8))
    if ruta:
        x = [coords[c][0] for c in ruta] + [coords[ruta[0]][0]]
        y = [coords[c][1] for c in ruta] + [coords[ruta[0]][1]]
        plt.plot(x, y, '-', linewidth=1.2, zorder=1)
    plt.scatter(coords[:,0], coords[:,1], c='red', s=30, zorder=2, edgecolors='black', linewidths=0.4)
    plt.title(f'Solución LP para {nombre_escenario}', fontsize=12)
    plt.xlabel('X'); plt.ylabel('Y'); plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    nombre_archivo = os.path.join(output_dir, f'sim_lp_{nombre_escenario.replace(".csv","")}.png')
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close()
    return nombre_archivo

def ejecutarLP(nombre_csv, timeLimit=600, msg=False):
    base = os.path.basename(nombre_csv).replace('.csv','')
    output_dir = os.path.join(RESULTADOS_DIR, base)
    asegurar_directorio(output_dir)
    coords = cargar_ciudades_csv(os.path.join('escenarios', os.path.basename(nombre_csv)))
    distancias = calcular_matriz_distancias(coords)
    resultado = solve_tsp_pulp(distancias, timeLimit=timeLimit, msg=msg)
    resultado['n_ciudades'] = len(coords)
    resultado['output_dir'] = output_dir
    if resultado['ruta']:
        resultado['figura'] = visualizar_ruta_lp(coords, resultado['ruta'], os.path.basename(nombre_csv), output_dir)
    return resultado

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python ResolverLP.py <archivo.csv>")
    else:
        archivo = sys.argv[1]
        print(f"Ejecutando LP sobre {archivo} (timeLimit 600s por defecto)")
        res = ejecutarLP(archivo, timeLimit=600, msg=True)
        print("Status:", res['status'])
        print("Valor objetivo:", res['valor'])
        print("Tiempo (s):", res['runtime'])
        print("n_vars:", res['n_vars'], " n_constraints:", res['n_constraints'])
        if 'figura' in res:
            print("Figura guardada en:", res['figura'])

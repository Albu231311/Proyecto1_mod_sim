import os
import time
import pandas as pd
from TSP import asegurar_directorio, RESULTADOS_DIR, ESCENARIOS_DIR
from TSP import ejecutarGA
from ResolverLP import ejecutarLP

COMPARATIVA_DIR = os.path.join(RESULTADOS_DIR, 'comparativa')
asegurar_directorio(COMPARATIVA_DIR)

def comparar_tres_escenarios(escenarios, ga_params, lp_timeLimit=600, k_repeticiones=5):
    filas = []
    for archivo in escenarios:
        print("\n" + "*-"*30)
        print(f"Procesando escenario: {archivo}")
        print("*-"*30)
        res_ga = ejecutarGA(archivo, ga_params['N'], ga_params['maxIter'], ga_params['pct_seleccion'], ga_params['pct_cruce'], ga_params['pct_mutacion'], seed=ga_params.get('seed', None), k_repeticiones=k_repeticiones, aplicar2Opt=True, verbose=True)
        res_lp = ejecutarLP(os.path.join(ESCENARIOS_DIR, archivo), timeLimit=lp_timeLimit, msg=False)
        lp_valor = res_lp.get('valor', None)
        lp_status = res_lp.get('status', None)
        filas.append({
            'Escenario': archivo,
            'Metodo': 'LP (MTZ)',
            'n_ciudades': res_ga['n_ciudades'],
            'GA_N': ga_params['N'],
            'GA_maxIter': ga_params['maxIter'],
            'LP_n_vars': res_lp.get('n_vars', None),
            'LP_n_constraints': res_lp.get('n_constraints', None),
            'LP_status': lp_status,
            'Tiempo_s': res_lp.get('runtime', None),
            'Distancia': lp_valor,
            'Ruta': res_lp.get('ruta', None),
            'Error_%': 0.0
        })
        error_pct = None
        if lp_valor is not None and lp_valor != 0:
            error_pct = (res_ga['distancia_media'] - lp_valor) / lp_valor * 100
        filas.append({
            'Escenario': archivo,
            'Metodo': f'GA (media {k_repeticiones} rep)',
            'n_ciudades': res_ga['n_ciudades'],
            'GA_N': ga_params['N'],
            'GA_maxIter': ga_params['maxIter'],
            'LP_n_vars': res_lp.get('n_vars', None),
            'LP_n_constraints': res_lp.get('n_constraints', None),
            'LP_status': lp_status,
            'Tiempo_s': res_ga['runtime_medio'],
            'Distancia': res_ga['distancia_media'],
            'Distancia_std': res_ga['distancia_std'],
            'Distancia_min': res_ga['distancia_min'],
            'Distancia_max': res_ga['distancia_max'],
            'Error_%': error_pct
        })
        for rep_idx, resultado in enumerate(res_ga['resultados'], start=1):
            top3 = resultado['top3']
            for i, (ruta_tuple, distancia) in enumerate(top3, start=1):
                error_pct = None
                if lp_valor is not None and lp_valor != 0 and distancia is not None:
                    error_pct = (distancia - lp_valor) / lp_valor * 100
                filas.append({
                    'Escenario': archivo,
                    'Metodo': f'GA_rep{rep_idx}_top{i}',
                    'n_ciudades': res_ga['n_ciudades'],
                    'GA_N': ga_params['N'],
                    'GA_maxIter': ga_params['maxIter'],
                    'LP_n_vars': res_lp.get('n_vars', None),
                    'LP_n_constraints': res_lp.get('n_constraints', None),
                    'LP_status': lp_status,
                    'Tiempo_s': resultado['runtime'],
                    'Distancia': distancia,
                    'Ruta': list(ruta_tuple),
                    'Error_%': error_pct,
                    'Seed': resultado['seed']
                })
    df = pd.DataFrame(filas)
    return df

if __name__ == "__main__":
    escenarios = [f for f in os.listdir(ESCENARIOS_DIR) if f.endswith('.csv')]
    escenarios.sort()
    ga_params = {'N': 100, 'maxIter': 500, 'pct_seleccion': 0.2, 'pct_cruce': 0.6, 'pct_mutacion': 0.2, 'seed': 42}
    print("Comparación automática GA vs LP para escenarios en carpeta 'escenarios/'")
    print("Se ejecutarán 5 repeticiones del GA por escenario (ajusta en el script).")
    df_res = comparar_tres_escenarios(escenarios, ga_params, lp_timeLimit=600, k_repeticiones=5)
    file_all = os.path.join(COMPARATIVA_DIR, 'tabla_comparativa.csv')
    file_res = os.path.join(COMPARATIVA_DIR, 'tabla_comparativa_resumen.csv')
    df_res.to_csv(file_all, index=False)
    cols_resumen = ['Escenario', 'Metodo', 'n_ciudades', 'Tiempo_s', 'Distancia', 'Error_%', 'LP_status']
    df_resumen = df_res[df_res['Metodo'].isin([f'GA (media 5 rep)', 'LP (MTZ)'])][cols_resumen]
    df_resumen.to_csv(file_res, index=False)
    print(f"\nTablas guardadas en: {COMPARATIVA_DIR}")
    print(df_resumen)
# Proyecto TSP: Algoritmos Genéticos vs Programación Lineal

> **Proyecto 1 - Modelación y Simulación 2025**  
> Comparación entre Algoritmos Genéticos y Optimización mediante OR-Tools para resolver el Problema del Vendedor Viajero (TSP)

Este proyecto implementa y compara **dos enfoques diferentes** para resolver el **Traveling Salesman Problem (TSP)**:

### Algoritmo Genético (GA)
Implementación de un algoritmo evolutivo que simula la selección natural para encontrar rutas óptimas. Incluye:
- Operadores de selección elitista
- Cruce Order Crossover (OX) especializado
- Mutación dual (Swap + Inversión)
- Visualización en tiempo real de la evolución
- Control completo de parámetros

### Programación Lineal (LP)
Solución mediante Google OR-Tools utilizando:
- Modelado como problema de routing
- Metaheurística Guided Local Search
- Garantía de soluciones near-óptimas
- Alta escalabilidad (200+ ciudades)

### Sistema de Comparación
Script automatizado que:
- Compara resultados de ambos métodos
- Genera tablas de rendimiento
- Calcula porcentajes de error
- Exporta resultados a CSV

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/Albu231311/Proyecto1_mod_sim.git
cd Proyecto1_mod_sim

pip install numpy pandas matplotlib ortools

# Ejecutar el algoritmo genético
python genetic_algorithm.py

# Ejecutar la programación lineal
python linear_programming.py

# Comparar ambos métodos
python comparison.py

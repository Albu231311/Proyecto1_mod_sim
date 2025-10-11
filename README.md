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

# Ejecuta en orden:
python TSP.py      # Primero
python Lineal.py   # Segundo
python comparacion.py  # Tercero

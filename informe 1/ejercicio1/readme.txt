Búsqueda A* en el Problema de Rumania
Descripción
Este proyecto implementa el algoritmo de búsqueda A* para encontrar la ruta más corta entre dos ciudades del mapa de Rumania.
Se utiliza como ejemplo clásico del libro Artificial Intelligence: A Modern Approach, en el que el objetivo es viajar desde una ciudad inicial hasta una ciudad objetivo minimizando la distancia total recorrida.

En este caso, la ciudad inicial es Arad y la ciudad objetivo es Bucharest.

Análisis del problema
El problema se modela como un grafo no dirigido en el que:

Nodos: ciudades de Rumania.

Aristas: carreteras entre ciudades con su respectivo costo en kilómetros.

Heurística: distancia en línea recta desde cada ciudad hasta la ciudad objetivo.

La tarea consiste en encontrar el camino más corto desde la ciudad inicial hasta la meta usando la información de las distancias reales y la estimación heurística.

Cómo se aplica A*
El algoritmo A* combina:

g(n): costo real acumulado desde la ciudad inicial hasta el nodo actual.

h(n): estimación heurística del costo restante hasta la ciudad objetivo.

f(n) = g(n) + h(n): estimación total del costo del camino que pasa por ese nodo.

Pasos del algoritmo:

Se comienza desde el nodo inicial (ciudad de partida).

Se expande el nodo actual generando sus vecinos.

Se calcula g(n), h(n) y f(n) para cada vecino.

Se selecciona siempre el nodo con menor f(n) para expandirlo en la siguiente iteración.

Cuando el nodo expandido es la ciudad objetivo, el camino encontrado es el óptimo.

En este proyecto se utiliza una cola de prioridad (heapq) para administrar los nodos según su f(n).

Por qué la ruta es óptima
El algoritmo A* encuentra la ruta óptima siempre que la heurística sea admisible, es decir, que nunca sobreestime el costo real restante.
En este caso, la heurística es la distancia en línea recta hasta la meta, la cual es siempre menor o igual a la distancia real por carretera.

Por esta razón, la primera vez que A* llega a la ciudad objetivo, la ruta encontrada es garantizadamente la de menor costo total.

Requisitos
Python 3.8 o superior.

ejecución



Ejecutar el script con:

bash

python rutaoptima2.py
Ejemplo de salida

Paso 1:
→ Expandiendo nodo: Arad, g(n)=0, h(n)=366, f(n)=366
   - Hijo generado: Zerind, g=75, h=374, f=449
   - Hijo generado: Sibiu, g=140, h=253, f=393
   - Hijo generado: Timisoara, g=118, h=329, f=447
...

=== SOLUCIÓN FINAL ===
Ruta óptima encontrada: ['Arad', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest']
Costo total: 418


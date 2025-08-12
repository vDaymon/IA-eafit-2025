Ejercicio 3 – Búsqueda en Grafo (BFS vs IDS)
1. Descripción del problema
Se modela una red de estaciones como un grafo no ponderado, donde cada nodo representa una estación y las aristas representan conexiones directas entre ellas.
El objetivo es encontrar la ruta más corta entre dos estaciones usando Búsqueda en Amplitud (BFS) y Búsqueda en Profundidad Iterativa (IDS), comparando su rendimiento.

2. Grafo
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B", "G"],
    "E": ["B", "H", "I"],
    "F": ["C", "J"],
    "G": ["D"],
    "H": ["E"],
    "I": ["E", "J"],
    "J": ["F", "I"]
}

Nodos: estaciones (A–J)
Aristas: conexiones directas (bidireccionales)
Costo: uniforme (1 por tramo)


3. Algoritmos utilizados
Búsqueda en Amplitud (BFS)

-Explora el grafo por niveles.
-Garantiza encontrar la solución óptima en grafos no ponderados.
-Mayor consumo de memoria, ya que almacena todos los nodos del nivel actual.
-Búsqueda en Profundidad Iterativa (IDS)
-Combina DFS con límites crecientes de profundidad.
-Uso de memoria reducido.
-Puede repetir exploraciones, aumentando el número total de nodos expandidos.

4. Conclusiones
BFS: Menor número de nodos expandidos y encuentra la ruta óptima rápidamente, pero consume más memoria.

IDS: Mayor número de nodos expandidos debido a exploraciones repetidas, pero con un uso de memoria menor.

En grafos pequeños y con soluciones poco profundas, BFS suele ser más eficiente.

En grafos muy grandes o con poca memoria disponible, IDS es una opción más segura.

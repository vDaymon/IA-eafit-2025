Maze Problem

1. Resolver el ejercicio planteado
El ejercicio consistía en implementar un algoritmo de búsqueda informada para encontrar la salida en un laberinto.
En este caso, se utilizó Greedy Best-First Search con heurística Manhattan, que prioriza expandir siempre el nodo que parece más cercano a la meta en términos de distancia ortogonal (sin paredes).

Funcionamiento básico:

Representamos el laberinto como una matriz con S (inicio), E (salida), # (paredes) y espacios vacíos.

Se define un objeto Problem con:

Acciones posibles (Up, Down, Left, Right).

Función result para calcular el nuevo estado al movernos.

Función action_cost (aquí cada paso cuesta 1).

El algoritmo mantiene:

Una frontera (frontier) que es una cola de prioridad ordenada por la heurística.

Un conjunto de visitados (reached) para evitar caminos repetidos peores.

El proceso continúa hasta llegar a la salida, reconstruyendo el camino recorrido.

Resultado: El robot encuentra un camino válido hacia la salida siguiendo la estimación más corta según la heurística.

2. ¿Cómo cambia el comportamiento si cambiamos la función de costo?
Por defecto, action_cost devuelve 1 para todos los movimientos, lo que hace que el algoritmo solo priorice la heurística (h(n)).

Si cambiamos el costo por ejemplo:

def action_cost(self, state, move):
    cell = self.maze[state[0] + move[0]][state[1] + move[1]]
    if cell == 'M':  # obstaculo
        return 5     # cuesta más pasar
    return 1

Entonces:

Con Greedy Best-First, el orden de exploración no cambia mucho, porque el algoritmo sigue decidiendo por la heurística, pero sí puede afectar el camino final si los nuevos costos impiden mejorar un estado ya visitado.

3. ¿Qué sucede si hay múltiples salidas? ¿Cómo modificar el algoritmo?
Si el laberinto tiene más de una salida, el algoritmo actual solo está configurado para un único goal.
Para manejar múltiples salidas:

-Guardamos todas las metas en una lista
-Modificamos la heurística para calcular la distancia mínima desde el estado actual a cualquiera de las salidas:
-En el bucle principal, el algoritmo termina si la posición actual está en goals, esto haría que el robot se dirija siempre hacia la salida más cercana según la heurística.

4. Laberinto más grande y otro tipo de obstáculo
Podemos ampliar el laberinto y añadir otros elementos además de paredes:

M (Obstaculos): transitable pero con mayor costo.
W (agua): no transitable.

En la carpeta hay dos codigos, el primero resuelve el problema base, el segundo es una pequeña modificacion, que acepta labernitos con varias salidas y mas obstaculos, a continuacion dejare las diferencias entre los dos codigos aunque la logica base sigue siendo la misma

- Comparación entre Código 1 y Código 2

Objetivo y alcance del problema

Código 1: Diseñado para un solo objetivo (goal) en el laberinto. Solo maneja paredes (#) como obstáculos y cada paso tiene un costo uniforme de 1. Usa la distancia Manhattan a un único punto como heurística.

Código 2: Soporta múltiples objetivos (goals) y un laberinto más variado con diferentes tipos de celdas: M (barro, mayor costo), W (agua, intransitable) y # (pared). Calcula la heurística como la distancia Manhattan a la meta más cercana.

Estructura de Problem

Código 1: Guarda un único goal. La función action_cost() siempre devuelve 1.

Código 2: Guarda una lista de goals. La función action_cost() ajusta el costo según el tipo de celda de destino (M cuesta 5, lo demás cuesta 1). Además, convierte siempre las posiciones a tuplas de enteros para evitar errores de tipos.

Manejo de la cola de prioridad (heapq)

Código 1: La cola de prioridad almacena pares (heurística, nodo). Si dos nodos tienen la misma heurística, Python intenta comparar nodos, lo que puede generar errores.

Código 2: La cola de prioridad almacena (heurística, contador, nodo), donde el contador se incrementa en cada inserción y evita que heapq compare nodos en caso de empate. Esto hace el código más estable.

Exploración de vecinos

Código 1: Solo revisa que la celda no sea pared (#). No valida si la nueva posición está fuera de los límites del laberinto.

Código 2: Primero verifica que la posición esté dentro de los límites de la matriz. Luego descarta paredes (#) y agua (W). También maneja correctamente las coordenadas para evitar operaciones entre enteros y cadenas.

Resumen:
El Código 1 es una implementación simple, adecuada para un único objetivo y un laberinto básico con costos uniformes.
El Código 2 es una versión más robusta y general, capaz de manejar múltiples metas, terrenos con costos diferentes, varios tipos de obstáculos y evitando errores en la cola de prioridad.

- Por ultimo la principal limitación del algoritmo que hemos usado es que:

-No considera bien terrenos con costos variables si la heurística no se ajusta
-La distancia Manhattan asume que cada paso cuesta lo mismo, pero si hay terrenos como M (costo 5), puede subestimar o sobrestimar el costo real, esto puede llevar a que explore rutas que parecen más cortas en distancia pero que son más caras en costo total.
-A medida que el laberinto crece, el número de nodos explorados crece rápidamente y el consumo de memoria aumenta.


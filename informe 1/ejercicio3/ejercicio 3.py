from collections import deque
import tracemalloc
import time

# --- Grafo de la red ---
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

# --- Clases ---
class Node:
    def __init__(self, state, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.depth = depth

    def path(self):
        node, p = self, []
        while node:
            p.append(node.state)
            node = node.parent
        return list(reversed(p))

class Problem:
    def __init__(self, graph, initial, goal):
        self.graph = graph
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        return list(self.graph.get(state, []))

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.goal

# --- BFS ---
def bfs(problem):
    start_time = time.perf_counter()
    tracemalloc.start()

    root = Node(problem.initial)
    if problem.goal_test(root.state):
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return root.path(), 0, 0, time.perf_counter()-start_time, peak

    frontier = deque([root])
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored and all(n.state != child_state for n in frontier):
                child = Node(child_state, parent=node, depth=node.depth+1)
                if problem.goal_test(child.state):
                    peak = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    return child.path(), nodes_expanded+1, child.depth, time.perf_counter()-start_time, peak
                frontier.append(child)
        nodes_expanded += 1

    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return None, nodes_expanded, None, time.perf_counter()-start_time, peak

# --- DLS para IDS ---
def dls(problem, limit):
    root = Node(problem.initial)
    nodes_expanded = 0
    visited_on_path = set()

    def recursive_dls(node, limit):
        nonlocal nodes_expanded
        nodes_expanded += 1
        if problem.goal_test(node.state):
            return ("solved", node)
        elif node.depth == limit:
            return ("cutoff", None)
        else:
            cutoff_occurred = False
            visited_on_path.add(node.state)
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state not in visited_on_path:
                    child = Node(child_state, parent=node, depth=node.depth+1)
                    result = recursive_dls(child, limit)
                    if result[0] == "cutoff":
                        cutoff_occurred = True
                    elif result[0] == "solved":
                        return result
            visited_on_path.discard(node.state)
            return ("cutoff", None) if cutoff_occurred else ("failure", None)

    return recursive_dls(root, limit), nodes_expanded

# --- IDS ---
def ids(problem, max_depth=50):
    start_time = time.perf_counter()
    tracemalloc.start()

    total_nodes_expanded = 0
    for depth in range(max_depth+1):
        (result, node), nodes = dls(problem, depth)
        total_nodes_expanded += nodes
        if result == "solved":
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return node.path(), total_nodes_expanded, node.depth, time.perf_counter()-start_time, peak, depth

    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return None, total_nodes_expanded, None, time.perf_counter()-start_time, peak, max_depth

# --- Ejecución ---
problem = Problem(graph, "A", "J")

bfs_path, bfs_nodes, bfs_depth, bfs_time, bfs_mem = bfs(problem)
ids_path, ids_nodes, ids_depth, ids_time, ids_mem, ids_limit = ids(problem, max_depth=10)

# --- Resultados ---
print("=== BFS ===")
print(f"Ruta: {bfs_path}")
print(f"Nodos expandidos: {bfs_nodes}")
print(f"Profundidad: {bfs_depth}")
print(f"Tiempo: {bfs_time:.6f} s")
print(f"Memoria pico: {bfs_mem} bytes\n")

print("=== IDS ===")
print(f"Ruta: {ids_path}")
print(f"Nodos expandidos: {ids_nodes}")
print(f"Profundidad: {ids_depth}")
print(f"Tiempo: {ids_time:.6f} s")
print(f"Memoria pico: {ids_mem} bytes")
print(f"Límite de profundidad usado: {ids_limit}")

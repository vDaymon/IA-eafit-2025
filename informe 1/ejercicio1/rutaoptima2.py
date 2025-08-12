import heapq

# ==== Clase Nodo ====
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

# ==== Expansión de nodos ====
def expand(problem, node):
    children = []
    for action in problem.actions(node.state):
        result_state = problem.result(node.state, action)
        cost = problem.action_cost(node.state, action, result_state)
        child = Node(
            state=result_state,
            parent=node,
            action=action,
            path_cost=node.path_cost + cost
        )
        children.append(child)
    return children

# ==== Clase Problema ====
class Problem:
    def __init__(self, initial, goal, actions, result, action_cost, is_goal):
        self.initial = initial
        self.goal = goal
        self.actions = actions
        self.result = result
        self.action_cost = action_cost
        self.is_goal = is_goal

# ==== Algoritmo A* con impresión paso a paso ====
def best_first_search(problem, f):
    node = Node(state=problem.initial)
    frontier = [(f(node), node)]
    heapq.heapify(frontier)
    reached = {problem.initial: node}

    paso = 1
    while frontier:
        _, node = heapq.heappop(frontier)
        print(f"\nPaso {paso}:")
        print(f"→ Expandiendo nodo: {node.state}, g(n)={node.path_cost}, h(n)={heuristicSLD[node.state]}, f(n)={f(node)}")
        paso += 1

        if problem.is_goal(node.state):
            return node

        for child in expand(problem, node):
            s = child.state
            print(f"   - Hijo generado: {s}, g={child.path_cost}, h={heuristicSLD[s]}, f={f(child)}")
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                heapq.heappush(frontier, (f(child), child))
    return None

# ==== Definiciones del problema de Rumania ====
initial = 'Arad'
goal = 'Bucharest'

heuristicSLD = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Dobreta': 242, 'Eforie': 161,
    'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151, 'Iasi': 226, 'Lugoj': 244,
    'Mehadia': 241, 'Neamt': 234, 'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193,
    'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

actions = {
    'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
    'Zerind': ['Arad', 'Oradea'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu Vilcea'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Mehadia': ['Lugoj', 'Dobreta'],
    'Dobreta': ['Mehadia', 'Craiova'],
    'Craiova': ['Dobreta', 'Rimnicu Vilcea', 'Pitesti'],
    'Rimnicu Vilcea': ['Sibiu', 'Craiova', 'Pitesti'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Pitesti': ['Rimnicu Vilcea', 'Craiova', 'Bucharest'],
    'Bucharest': ['Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni'],
    'Giurgiu': ['Bucharest'],
    'Urziceni': ['Bucharest', 'Vaslui', 'Hirsova'],
    'Hirsova': ['Urziceni', 'Eforie'],
    'Eforie': ['Hirsova'],
    'Vaslui': ['Urziceni', 'Iasi'],
    'Iasi': ['Vaslui', 'Neamt'],
    'Neamt': ['Iasi']
}

action_costs = {
    ('Arad', 'Zerind'): 75, ('Zerind', 'Arad'): 75,
    ('Arad', 'Sibiu'): 140, ('Sibiu', 'Arad'): 140,
    ('Arad', 'Timisoara'): 118, ('Timisoara', 'Arad'): 118,
    ('Zerind', 'Oradea'): 71, ('Oradea', 'Zerind'): 71,
    ('Oradea', 'Sibiu'): 151, ('Sibiu', 'Oradea'): 151,
    ('Sibiu', 'Fagaras'): 99, ('Fagaras', 'Sibiu'): 99,
    ('Sibiu', 'Rimnicu Vilcea'): 80, ('Rimnicu Vilcea', 'Sibiu'): 80,
    ('Timisoara', 'Lugoj'): 111, ('Lugoj', 'Timisoara'): 111,
    ('Lugoj', 'Mehadia'): 70, ('Mehadia', 'Lugoj'): 70,
    ('Mehadia', 'Dobreta'): 75, ('Dobreta', 'Mehadia'): 75,
    ('Dobreta', 'Craiova'): 120, ('Craiova', 'Dobreta'): 120,
    ('Craiova', 'Rimnicu Vilcea'): 146, ('Rimnicu Vilcea', 'Craiova'): 146,
    ('Craiova', 'Pitesti'): 138, ('Pitesti', 'Craiova'): 138,
    ('Rimnicu Vilcea', 'Pitesti'): 97, ('Pitesti', 'Rimnicu Vilcea'): 97,
    ('Fagaras', 'Bucharest'): 211, ('Bucharest', 'Fagaras'): 211,
    ('Pitesti', 'Bucharest'): 101, ('Bucharest', 'Pitesti'): 101,
    ('Bucharest', 'Giurgiu'): 90, ('Giurgiu', 'Bucharest'): 90,
    ('Bucharest', 'Urziceni'): 85, ('Urziceni', 'Bucharest'): 85,
    ('Urziceni', 'Hirsova'): 98, ('Hirsova', 'Urziceni'): 98,
    ('Hirsova', 'Eforie'): 86, ('Eforie', 'Hirsova'): 86,
    ('Urziceni', 'Vaslui'): 142, ('Vaslui', 'Urziceni'): 142,
    ('Vaslui', 'Iasi'): 92, ('Iasi', 'Vaslui'): 92,
    ('Iasi', 'Neamt'): 87, ('Neamt', 'Iasi'): 87
}

def result(state, action):
    return action

def action_cost(state, action, result):
    return action_costs.get((state, action), float('inf'))

def is_goal(state):
    return state == goal

# ==== f(n) para A* ====
def f(node):
    return node.path_cost + heuristicSLD[node.state]

# ==== Ejecución ====
problem = Problem(initial, goal, lambda s: actions.get(s, []), result, action_cost, is_goal)
solution = best_first_search(problem, f)

if solution:
    path = []
    total_cost = solution.path_cost
    while solution:
        path.append(solution.state)
        solution = solution.parent
    path.reverse()
    print("\n=== SOLUCIÓN FINAL ===")
    print("Ruta óptima encontrada:", path)
    print("Costo total:", total_cost)
else:
    print("No se encontró solución")

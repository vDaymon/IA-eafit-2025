import heapq

class Node:
    def __init__(self, position, parent=None, action=None, path_cost=0):
        self.position = position      # (x, y)
        self.parent = parent          # nodo padre
        self.action = action          # acción tomada para llegar aquí
        self.path_cost = path_cost    # costo acumulado

    def __lt__(self, other):
        return self.path_cost < other.path_cost


class Problem:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        # Movimientos posibles (x, y) : nombre
        self.actions = {
            (-1, 0): "Up",
            (1, 0): "Down",
            (0, -1): "Left",
            (0, 1): "Right"
        }

    def result(self, state, move):
        return (state[0] + move[0], state[1] + move[1])

    def action_cost(self, state, move):
        return 1


def find_exit(maze):
    start = (1, 1)
    end = (1, 6)
    problem = Problem(maze, start, end)

    def manhattan_distance(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(node):
        neighbors = []
        for move, action_name in problem.actions.items():
            new_pos = problem.result(node.position, move)
            # verificar que no sea pared
            if maze[new_pos[0]][new_pos[1]] != "#":
                neighbors.append((new_pos, action_name))
        return neighbors

    start_node = Node(start, path_cost=0)
    frontier = [(manhattan_distance(start, end), start_node)]
    heapq.heapify(frontier)
    reached = {start: start_node}

    while frontier:
        _, node = heapq.heappop(frontier)
        if node.position == end:
            return reconstruct_path(node)

        for new_pos, action_name in get_neighbors(node):
            new_cost = node.path_cost + problem.action_cost(node.position, action_name)
            if new_pos not in reached or new_cost < reached[new_pos].path_cost:
                reached[new_pos] = Node(new_pos, parent=node, action=action_name, path_cost=new_cost)
                heapq.heappush(frontier, (manhattan_distance(new_pos, end), reached[new_pos]))

    return None


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.position, node.action))
        node = node.parent
    path.reverse()
    return path[1:]  # eliminamos el primero porque no tiene acción


# Laberinto
maze = [
    ["#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", "S", "#", " ", "#", " ", "E", "#"],
    ["#", " ", " ", " ", "#", " ", " ", "#"],
    ["#", " ", "#", " ", " ", " ", "#", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#"]
]

# Ejecutar
path = find_exit(maze)

print("Path to exit:")
for pos, action in path:
    print(f"{action} -> {pos}")
    



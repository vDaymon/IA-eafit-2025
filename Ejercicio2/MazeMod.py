import heapq

class Node:
    def __init__(self, position, parent=None, action=None, path_cost=0):
        self.position = tuple(position)  # aseguramos que sea tupla de enteros
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost


class Problem:
    def __init__(self, maze, start, goals):
        self.maze = maze
        self.start = tuple(start)
        if isinstance(goals, tuple):
            self.goals = [tuple(goals)]
        else:
            self.goals = [tuple(g) for g in goals]
        self.actions = {
            (-1, 0): "Up",
            (1, 0): "Down",
            (0, -1): "Left",
            (0, 1): "Right"
        }

    def result(self, state, move):
        return (state[0] + move[0], state[1] + move[1])

    def action_cost(self, state, move):
        new_pos = self.result(state, move)
        cell = self.maze[new_pos[0]][new_pos[1]]
        if cell == 'M':  # Obstaculos
            return 5
        return 1


def manhattan_distance(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def min_manhattan(pos, goals):
    return min(manhattan_distance(pos, g) for g in goals)


def find_exit(maze, start, goals):
    problem = Problem(maze, start, goals)

    def get_neighbors(node):
        neighbors = []
        for move, action_name in problem.actions.items():
            new_pos = problem.result(node.position, move)
            # Comprobar límites
            if not (0 <= new_pos[0] < len(maze) and 0 <= new_pos[1] < len(maze[0])):
                continue
            # Evitar paredes y agua
            if maze[new_pos[0]][new_pos[1]] not in ("#", "W"):
                neighbors.append((new_pos, action_name))
        return neighbors

    start_node = Node(start, path_cost=0)
    counter = 0
    frontier = [(min_manhattan(start, problem.goals), counter, start_node)]
    heapq.heapify(frontier)
    reached = {start: start_node}

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node.position in problem.goals:
            return reconstruct_path(node)

        for new_pos, action_name in get_neighbors(node):
            new_cost = node.path_cost + problem.action_cost(node.position, (new_pos[0]-node.position[0], new_pos[1]-node.position[1]))
            if new_pos not in reached or new_cost < reached[new_pos].path_cost:
                reached[new_pos] = Node(new_pos, parent=node, action=action_name, path_cost=new_cost)
                counter += 1
                heapq.heappush(frontier, (min_manhattan(new_pos, problem.goals), counter, reached[new_pos]))

    return None


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.position, node.action))
        node = node.parent
    path.reverse()
    return path[1:]


if __name__ == "__main__":
    maze = [
        ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
        ["#", "S", " ", " ", "#", " ", "M", " ", "E", "#"],
        ["#", " ", "#", " ", "#", " ", "W", " ", "#", "#"],
        ["#", " ", "M", " ", " ", " ", "#", " ", " ", "#"],
        ["#", "#", "#", "W", "#", "#", "#", "#", "E", "#"],
        ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#"]
    ]

    start = (1, 1)
    goals = [(1, 8), (4, 8)]

    path = find_exit(maze, start, goals)

    print("Path to exit:")
    if path:
        for pos, action in path:
            print(f"{action} -> {pos}")
    else:
        print("No path found")


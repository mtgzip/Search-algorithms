import heapq
from math import sqrt

WALL = "X"
START_STATE = "S"
GOAL_STATE = "G"


def plan(map, algorithm="bfs", heuristic=None):
    """Loads a level, searches for a path between the given waypoints, and displays the result.

    Args:
        filename: The name of the text file containing the level.
        src_waypoint: The character associated with the initial waypoint.
        dst_waypoint: The character associated with the destination waypoint.

    """
    print(map)
    print("Algorithm:", algorithm)
    print("Heuristic:", heuristic)

    # Load the level from the file
    level = parse_level(map)

    # Retrieve the source and destination coordinates from the level.
    start = level["start"]
    goal = level["goal"]

    # Search for and display the path from src to dst.
    path = []
    visited = {}

    if algorithm == "bfs":
        path, visited = bfs(start, goal, level, transition_model)
    elif algorithm == "dfs":
        path, visited = dfs(start, goal, level, transition_model)
    elif algorithm == "ucs":
        path, visited = ucs(start, goal, level, transition_model)
    elif algorithm == "greedy":
        if heuristic == "euclidian":
            path, visited = greedy_best_first(
                start, goal, level, transition_model, h_euclidian
            )
        elif heuristic == "manhattan":
            path, visited = greedy_best_first(
                start, goal, level, transition_model, h_manhattan
            )
    elif algorithm == "astar":
        if heuristic == "euclidian":
            path, visited = a_star(start, goal, level, transition_model, h_euclidian)
        elif heuristic == "manhattan":
            path, visited = a_star(start, goal, level, transition_model, h_manhattan)

    return path, path_cost(path, level), visited


def parse_level(map):
    """Parses a level from a string.

    Args:
        level_str: A string containing a level.

    Returns:
        The parsed level (dict) containing the locations of walls (set), the locations of spaces
        (dict), and a mapping of locations to waypoints (dict).
    """
    start = None
    goal = None
    walls = set()
    spaces = {}

    for j, line in enumerate(map.split("\n")):
        for i, char in enumerate(line):
            if char == "\n":
                continue
            elif char == WALL:
                walls.add((i, j))
            elif char == START_STATE:
                start = (i, j)
                spaces[(i, j)] = 1.0
            elif char == GOAL_STATE:
                goal = (i, j)
                spaces[(i, j)] = 1.0
            elif char.isnumeric():
                spaces[(i, j)] = float(char)

    level = {"walls": walls, "spaces": spaces, "start": start, "goal": goal}
    print(level)
    return level


def path_cost(path, level):
    """Returns the cost of the given path.

    Args:
        path: A list of cells from the source to the goal.
        level: A loaded level, containing walls, spaces, and waypoints.

    Returns:
        The cost of the given path.
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += cost_function(
            level,
            path[i],
            path[i + 1],
            level["spaces"][path[i]],
            level["spaces"][path[i + 1]],
        )

    return cost


# =============================
# Transition Model
# =============================


def cost_function(level, state1, state2, cost1, cost2):
    """Returns the cost of the edge joining state1 and state2.

    Args:
        state1: A source location.
        state2: A target location.

    Returns:
        The cost of the edge joining state1 and state2.
    """

    def dist(state1, state2):
        euclidean = sqrt((state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2)
        return euclidean

    cost = dist(state1, state2) * (cost1 + cost2) / 2

    return cost


def transition_model(level, state1):
    """Provides a list of adjacent states and their respective costs from the given state.

    Args:
        level: A loaded level, containing walls, spaces, and waypoints.
        state: A target location.

    Returns:
        A list of tuples containing an adjacent sates's coordinates and the cost of
        the edge joining it and the originating state.

        E.g. from (0,0):
            [((0,1), 1),
             ((1,0), 1),
             ((1,1), 1.4142135623730951),
             ... ]
    """
    adj_states = {}
    x, y = state1
    movimentos = [
        (-1, 0),
        (-1, -1),
        (0, -1),
        (+1, -1),
        (1, 0),
        (+1, +1),
        (0, +1),
        (-1, +1),
    ]
    # O custo para o state1 é buscado em level['spaces'], assumindo que sempre existe
    cost1 = level["spaces"].get(
        state1, 1
    )  # aqui buscamos o valor na chave de spaces, caso não existir assumimos 1
    for dx, dy in movimentos:
        novo_x, novo_y = x + dx, y + dy
        novo_estado = (novo_x, novo_y)  # aqui criamos os novos possiveis estados

        # Verifica se o novo estado não é uma parede e está dentro dos espaços disponíveis
        if novo_estado not in level["walls"] and novo_estado in level["spaces"]:
            cost2 = level["spaces"][novo_estado]  # Obtém o custo para o state2
            # Calcula o custo usando a cost_function
            custo = cost_function(level, state1, novo_estado, cost1, cost2)
            adj_states[novo_estado] = (
                custo  # atribuimos o custo ao estado novo adjacente
            )

    return adj_states.items()


# =============================
# Uninformed Search Algorithms
# =============================


def bfs(s, g, level, adj):
    """Realiza uma busca em largura para encontrar um caminho do estado inicial ao estado final.

    Args:
        s: O estado inicial.
        g: O estado final.
        level: O mapa representado pelo dicionário level.
        adj: A função de transição.

    Returns:
        Uma lista path com o caminho entre s e g encontrado pela busca em largura e um dicionário visited 
        contendo todos os estados visitados durante o processo.
    """
    visited = {s: None}  # Dicionário para manter o mapeamento dos estados visitados e seus pais
    fila = [s]  # Fila para armazenar os estados a serem explorados
    alcancado = {s}  # Conjunto para evitar ciclos

    while fila:
        n = fila.pop(0)  # Remove o primeiro estado da fila
        if n == g:
            break
        # Obtém os estados adjacentes ao estado atual usando a função de transição
        adjacent_states = adj(level, n) # Usa a função para mapear os proximos estados,
        for filho, _ in adjacent_states:  # Itera sobre os estados adjacentes, _ por que não usa o custo do caminho
            if filho not in alcancado:  # Verifica se o estado adjacente já foi visitado
                fila.append(filho)  # Adiciona o estado adjacente à fila
                visited[filho] = n  # Registra o estado adjacente como visitado e seu pai
                alcancado.add(filho)  # Adiciona o estado adjacente ao conjunto de estados alcançados
    # Se o estado final não foi alcançado, retorna uma lista vazia
    if g not in visited:
        return [], visited
    # Reconstrói o caminho percorrido, retrocedendo do estado final até o estado inicial
    path = []
    n = g # aqui reconstruimos o caminho de volta partindo de G a S
    while n != s:
        path.append(n)
        n = visited[n]
    path.append(s) # É uma lista de G a S com as tuplas do caminho
    path.reverse() # É a mesma lista porém agora de G a S
    # Verifica se o estado final foi alcançado
    if path[-1] != g:
        return [], visited  # Retorna uma lista vazia se o estado final não foi alcançado

    return path, visited



def dfs(s, g, level, adj):
    """ Searches for a path from the source to the goal using the Depth-First Search algorithm.
    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
    
    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}  # Dicionário para manter o mapeamento dos estados visitados e seus pais
    pilha = [s]  # Fila para armazenar os estados a serem explorados
    alcancado = {s}  # Conjunto para evitar ciclos

    while pilha:
        n = pilha.pop()  # Remove o ultimo estado da fila
        if n == g:
            break
        # Obtém os estados adjacentes ao estado atual usando a função de transição
        adjacent_states = adj(level, n) # Usa a função para mapear os proximos estados,
        for filho, _ in adjacent_states:  # Itera sobre os estados adjacentes, _ por que não usa o custo do caminho
            if filho not in alcancado:  # Verifica se o estado adjacente já foi visitado
                pilha.append(filho)  # Adiciona o estado adjacente à fila
                visited[filho] = n  # Registra o estado adjacente como visitado e seu pai
                alcancado.add(filho)  # Adiciona o estado adjacente ao conjunto de estados alcançados
    # Se o estado final não foi alcançado, retorna uma lista vazia
    if g not in visited:
        return [], visited
    # Reconstrói o caminho percorrido, retrocedendo do estado final até o estado inicial
    path = []
    n = g # aqui reconstruimos o caminho de volta partindo de G a S
    while n != s:
        path.append(n)
        n = visited[n]
    path.append(s) # É uma lista de G a S com as tuplas do caminho
    path.reverse() # É a mesma lista porém agora de G a S

    return path, visited



def ucs(s, g, level, adj):
    """Realiza uma busca de custo uniforme para encontrar um caminho do estado inicial ao estado final.

    Args:
        s: O estado inicial.
        g: O estado final.
        level: O mapa representado pelo dicionário level.
        adj: A função de transição.

    Returns:
        Uma lista path com o caminho entre s e g encontrado pela busca de custo uniforme e um dicionário visited
        contendo todos os estados visitados durante o processo.
    """
    visited = {s: None}  # Dicionário para manter o mapeamento dos estados visitados e seus pais
    fronteira = [(0, s)]  # Fila de prioridade para armazenar os estados a serem explorados
    alcancado = {s}  # Dicionário para evitar ciclos
    custo= {s: 0}
    while fronteira:
        custo_atual, n = heapq.heappop(fronteira)  # Remove o estado com menor custo da fronteira
        if n == g:
            break
        # Obtém os estados adjacentes ao estado atual usando a função de transição
        adjacent_states = adj(level, n) # mapeia os possiveis locais proximos
        for filho, cost in adjacent_states: # acessa as coordenadas e os custos referentes
            custo_filho = custo_atual + cost 
            if filho not in alcancado or custo_filho < custo[filho]:
                heapq.heappush(fronteira, (custo_filho, filho))
                visited[filho] = n
                alcancado.add(filho)
                custo[filho] = custo_filho

    # Reconstrói o caminho percorrido, retrocedendo do estado final até o estado inicial
    path = []
    n = g
    while n != s:
        path.append(n)
        n = visited[n]
    path.append(s)
    path.reverse()

    return path, visited



# ======================================
# Informed (Heuristic) Search Algorithms
# ======================================


def greedy_best_first(s, g, level, adj, h):
    """Searches for a path from the source to the goal using the Greedy Best-First Search algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
        h: A heuristic function that estimates the cost from the current cell to the goal.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}  # Dicionário para manter o mapeamento dos estados visitados e seus pais
    fronteira = [(0, s)]  # Fila de prioridade para armazenar os estados a serem explorados
    alcancado = {s}  # Dicionário para evitar ciclos e armazenar o custo mínimo para alcançar um estado
    custo= {s: 0}
    while fronteira:
        custo_filho, n = heapq.heappop(fronteira)  # Remove o estado com menor custo da fronteira
        if n == g:
            break
        # Obtém os estados adjacentes ao estado atual usando a função de transição
        adjacent_states = adj(level, n) # mapeia os possiveis locais proximos 
        for filho, _ in adjacent_states: # acessa as coordenadas e os custos referentes
            custo_filho = h(filho, g)

            if filho not in alcancado or custo_filho < custo[filho]:
                heapq.heappush(fronteira, (custo_filho, filho))
                visited[filho] = n
                alcancado.add(filho)
                custo[filho] = custo_filho
             
    # Se o estado final não foi alcançado, retorna uma lista vazia
    if g not in visited:
        return [], visited
    # Reconstrói o caminho percorrido, retrocedendo do estado final até o estado inicial
    path = []
    n = g
    while n != s:
        path.append(n)
        n = visited[n]
    path.append(s)
    path.reverse()

    return path, visited


def a_star(s, g, level, adj, h):
    """Searches for a path from the source to the goal using the A* algorithm.

    Args:
        s: The source location.
        g: The goal location.
        level: The level containing the locations of walls, spaces, and waypoints.
        adj: A function that returns the adjacent cells and their respective costs from the given cell.
        h: A heuristic function that estimates the cost from the current cell to the goal.

    Returns:
        A list of tuples containing cells from the source to the goal, and a dictionary containing the visited cells and their respective parent cells.
    """
    visited = {s: None}  # Dicionário para manter o mapeamento dos estados visitados e seus pais
    fronteira = [(0, s)]  # Fila de prioridade para armazenar os estados a serem explorados
    alcancado = {s}  # Dicionário para evitar ciclos e armazenar o custo mínimo para alcançar um estado
    custo= {s: 0}
    while fronteira:
        custo_f, n = heapq.heappop(fronteira)  # Remove o estado com menor custo da fronteira
        if n == g:
            break
        # Obtém os estados adjacentes ao estado atual usando a função de transição
        adjacent_states = adj(level, n) # mapeia os possiveis locais proximos 

        for filho, cost in adjacent_states: # acessa as coordenadas e os custos referentes
            custo_g = custo[n] + cost 
            cost_h=h(filho,g)
            custo_f= custo_g + cost_h
            if filho not in alcancado or custo_f < custo[filho]:
                heapq.heappush(fronteira, (custo_f, filho))
                visited[filho] = n
                alcancado.add(filho)
                custo[filho] = custo_g

    # Se o estado final não foi alcançado, retorna uma lista vazia
    if g not in visited:
        return [], visited
    # Reconstrói o caminho percorrido, retrocedendo do estado final até o estado inicial
    path = []
    n = g
    while n != s:
        path.append(n)
        n = visited[n]
    path.append(s)
    path.reverse()

    return path, visited

# ======================================
# Heuristic functions
# ======================================
def h_euclidian(s, g):
    """Estimates the cost from the current cell to the goal using the Euclidian distance.

    Args:
        s: The current location.
        g: The goal location.

    Returns:
        The estimated cost from the current cell to the goal.
    """

    xn, yn = s
    xg, yg = g
    euclidean = sqrt((xn - xg) ** 2 + (yn - yg) ** 2) #calculado igual na função de custo
    return euclidean


def h_manhattan(s, g):
    """Estimates the cost from the current cell to the goal using the Manhattan distance.

    Args:
        s: The current location.
        g: The goal location.

    Returns:
        The estimated cost from the current cell to the goal.
    """

    xn, yn = s
    xg, yg = g
    manhattan = abs(xn - xg) + abs(yn - yg) # usamos abs para ter o valor modulo da expressão 

    return manhattan

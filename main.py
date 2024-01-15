import heapq


def read_matrices(file_path):
    adjacency_matrix = []
    bandwidth_matrix = []
    delay_matrix = []
    reliability_matrix = []

    try:
        with open(file_path, 'r') as file:
            current_matrix = adjacency_matrix  # Initialize with a default matrix

            for line in file:
                line = line.strip()

                if not line:  # Empty line, switch to the next matrix
                    current_matrix = adjacency_matrix
                    continue

                if line.startswith("Adjacency:"):
                    current_matrix = adjacency_matrix
                elif line.startswith("Bandwidth:"):
                    current_matrix = bandwidth_matrix
                elif line.startswith("Delay:"):
                    current_matrix = delay_matrix
                elif line.startswith("Reliability:"):
                    current_matrix = reliability_matrix
                else:
                    values = list(map(float, line.split(':')))
                    if all(val.is_integer() for val in values):
                        values = list(map(int, values))
                    current_matrix.append(values)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix


def read_requests(file_path):
    """
    Read requests from the specified file.

    Parameters:
    - file_path (str): The path to the file containing requests.

    Returns:
    - List of tuples: Each tuple represents a request (source, destination, bandwidth, delay_threshold, reliability_threshold).
    """
    requests = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                values = list(map(float, line.strip().split()))
                if all(val.is_integer() for val in values):
                    values = list(map(int, values))

                if len(values) < 3:
                    print(f"Ignoring invalid line: {line}")
                    continue

                source, destination, bandwidth = map(int, values[:3])

                delay_threshold = 0.0
                reliability_threshold = 0.0

                if len(values) > 3:
                    delay_threshold = values[3]

                if len(values) > 4:
                    reliability_threshold = values[4]

                requests.append((source, destination, bandwidth, delay_threshold, reliability_threshold))

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return requests


def convert_to_adjacency_list(adjacency_matrix):
    """
    Convert adjacency matrix to adjacency list.

    Parameters:
    - adjacency_matrix (List[List[int]]): The adjacency matrix.

    Returns:
    - Dict: The adjacency list.
    """
    adjacency_list = {}

    for i, weights in enumerate(adjacency_matrix):
        neighbors = {}
        for j, weight in enumerate(weights):
            if weight > 0:
                neighbors[j] = weight
        adjacency_list[i] = neighbors

    return adjacency_list


def dijkstra_algorithm(adjacency_list, start, bandwidth_constraint):
    """
    Find the shortest paths from a source vertex to all other vertices using Dijkstra's algorithm.

    Parameters:
    - adjacency_list (Dict): The adjacency list.
    - start (int): The source vertex.
    - bandwidth_constraint (int): The bandwidth constraint.

    Returns:
    - Tuple of Dicts: distances and paths. distances[vertex] is the distance from the source to vertex.
      paths[vertex] is the path from the source to vertex.
    """
    distances = {vertex: float('infinity') for vertex in adjacency_list}
    distances[start] = 0
    priority_queue = [(0, start)]

    paths = {vertex: [] for vertex in adjacency_list}

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in adjacency_list[current_vertex].items():
            if weight <= bandwidth_constraint:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    paths[neighbor] = paths[current_vertex] + [current_vertex]

    return distances, paths


def bellman_ford_algorithm(adjacency_list, start, bandwidth_constraint):
    distances = {vertex: float('infinity') for vertex in adjacency_list}
    distances[start] = 0

    for _ in range(len(adjacency_list) - 1):
        for current_vertex, neighbors in adjacency_list.items():
            for neighbor, weight in neighbors.items():
                if weight <= bandwidth_constraint:
                    distance = distances[current_vertex] + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance

    paths = {vertex: [] for vertex in adjacency_list}
    return distances, paths


def a_star_algorithm(adjacency_list, start, goal, bandwidth_constraint, heuristic=None):
    distances = {vertex: float('infinity') for vertex in adjacency_list}
    distances[start] = 0
    priority_queue = [(0, start)]

    paths = {vertex: [] for vertex in adjacency_list}

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_vertex == goal:
            break  # Goal reached, no need to explore further

        for neighbor, weight in adjacency_list[current_vertex].items():
            if weight <= bandwidth_constraint:
                distance = distances[current_vertex] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    if heuristic:
                        priority = distance + heuristic(neighbor, goal)  # A* uses a heuristic
                    else:
                        priority = distance
                    heapq.heappush(priority_queue, (priority, neighbor))
                    paths[neighbor] = paths[current_vertex] + [current_vertex]

    return distances, paths


def heuristic(current, goal):
    return 0


def handle_request(adjacency_list, delay_matrix, reliability_matrix, request):
    source, destination, bandwidth_constraint, delay_threshold, reliability_threshold = request
    distances, paths = a_star_algorithm(adjacency_list, source, destination, bandwidth_constraint, heuristic)

    valid_paths = []

    for dest, path in paths.items():
        if dest == destination:
            for node in path:
                delay = delay_matrix[node][dest]
                reliability = reliability_matrix[node][dest]

                if delay <= delay_threshold and reliability >= reliability_threshold:
                    valid_paths.append(path)
                    break

    return valid_paths


def choose_algorithm(algorithm_name):
    if algorithm_name == "dijkstra":
        return dijkstra_algorithm
    elif algorithm_name == "bellman_ford":
        return bellman_ford_algorithm
    elif algorithm_name == "a_star":
        return a_star_algorithm


def simple_scheduling_algorithm(requests):
    return requests


def main():
    request_file_path = r"D:\algorithm\requests.txt"

    adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix = read_matrices(r"D:\algorithm\matrices.txt")

    adjacency_list = convert_to_adjacency_list(adjacency_matrix)

    requests = read_requests(request_file_path)

    algorithm_names = ["dijkstra", "bellman_ford", "a_star"]

    for algorithm_name in algorithm_names:
        algorithm = choose_algorithm(algorithm_name)

        if algorithm_name == "a_star":
            schedule = algorithm(adjacency_list, 0, 5, 39, heuristic)  # Adjust parameters as needed
        else:
            schedule = algorithm(adjacency_list, 0, 5)

        # Print the resulting schedule
        print(f"{algorithm_name.capitalize()} Algorithm Results:")
        distances, paths = schedule
        print("Distances:", distances)
        print("Paths:", paths)

    final_schedule = simple_scheduling_algorithm(requests)

    print("Final Schedule:")
    for entry in final_schedule:
        print(entry)


if __name__ == "__main__":
    main()
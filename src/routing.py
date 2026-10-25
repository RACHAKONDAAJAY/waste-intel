
import math
import random


def haversine_distance(point1, point2):
    """
    Calculate straight-line GPS distance between two coordinates.
    Coordinates must be (latitude, longitude).
    Returns distance in kilometres.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2

    earth_radius_km = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))
    c = 2 * math.asin(math.sqrt(a))

    return earth_radius_km * c


def create_distance_matrix(coordinates):
    """Create a matrix of distances between collection points."""
    number_of_points = len(coordinates)

    distance_matrix = [
        [0.0] * number_of_points
        for _ in range(number_of_points)
    ]

    for i in range(number_of_points):
        for j in range(i + 1, number_of_points):
            distance = haversine_distance(
                coordinates[i], coordinates[j]
            )
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


def calculate_route_distance(route, coordinates):
    """
    Calculate the closed-loop distance of a route.
    The route returns to its starting point.
    """
    if len(route) <= 1:
        return 0.0

    distance_matrix = create_distance_matrix(coordinates)
    total_distance = 0.0

    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]

    # Return to the starting point
    total_distance += distance_matrix[route[-1]][route[0]]

    return float(total_distance)


def tournament_selection(population, fitness_values, tournament_size=3):
    """Select a route using tournament selection."""
    tournament_size = min(tournament_size, len(population))

    selected_indices = random.sample(
        range(len(population)), tournament_size
    )

    winner_index = min(
        selected_indices,
        key=lambda index: fitness_values[index]
    )

    return population[winner_index][:]


def order_crossover(parent1, parent2):
    """Create a child route using order crossover."""
    number_of_points = len(parent1)

    if number_of_points < 3:
        return parent1[:]

    start, end = sorted(
        random.sample(range(number_of_points), 2)
    )

    child = [None] * number_of_points
    child[start:end + 1] = parent1[start:end + 1]

    remaining_points = [
        point for point in parent2 if point not in child
    ]

    remaining_index = 0

    for i in range(number_of_points):
        if child[i] is None:
            child[i] = remaining_points[remaining_index]
            remaining_index += 1

    return child


def mutate_route(route, mutation_rate=0.15):
    """Mutate a route by swapping two positions."""
    mutated_route = route[:]

    # Keep point 0 as the starting depot.
    if len(mutated_route) > 2 and random.random() < mutation_rate:
        first_index, second_index = random.sample(
            range(1, len(mutated_route)), 2
        )

        mutated_route[first_index], mutated_route[second_index] = (
            mutated_route[second_index],
            mutated_route[first_index],
        )

    return mutated_route


def genetic_algorithm_route(
    coordinates,
    population_size=80,
    generations=250,
    mutation_rate=0.15,
    seed=42,
):
    """
    Find a short collection route using a Genetic Algorithm.

    Point index 0 is kept as the starting depot.
    Returns (best_route, best_distance_km).
    """
    number_of_points = len(coordinates)

    if number_of_points == 0:
        raise ValueError("Add at least one collection point.")

    if number_of_points == 1:
        return [0], 0.0

    if population_size < 2:
        raise ValueError("Population size must be at least 2.")

    if generations < 1:
        raise ValueError("Generations must be at least 1.")

    if not 0 <= mutation_rate <= 1:
        raise ValueError("Mutation rate must be between 0 and 1.")

    random_generator = random.Random(seed)

    # Create initial routes, preserving depot index 0.
    population = []

    for _ in range(population_size):
        remaining_points = list(range(1, number_of_points))
        random_generator.shuffle(remaining_points)
        population.append([0] + remaining_points)

    distance_matrix = create_distance_matrix(coordinates)

    def route_distance(route):
        if len(route) <= 1:
            return 0.0

        distance = sum(
            distance_matrix[route[i]][route[i + 1]]
            for i in range(len(route) - 1)
        )

        return distance + distance_matrix[route[-1]][route[0]]

    best_route = list(range(number_of_points))
    best_distance = route_distance(best_route)

    # Include the original order in the population.
    population[0] = list(range(number_of_points))

    for _ in range(generations):
        fitness_values = [
            route_distance(route) for route in population
        ]

        generation_best_index = min(
            range(len(population)),
            key=lambda index: fitness_values[index]
        )

        if fitness_values[generation_best_index] < best_distance:
            best_route = population[generation_best_index][:]
            best_distance = fitness_values[generation_best_index]

        # Preserve the best route (elitism).
        new_population = [best_route[:]]

        while len(new_population) < population_size:
            # Select parents with a seeded random generator.
            def select_parent():
                candidates = random_generator.sample(
                    range(len(population)),
                    min(3, len(population)),
                )
                winner = min(
                    candidates,
                    key=lambda index: fitness_values[index],
                )
                return population[winner][:]

            parent1 = select_parent()
            parent2 = select_parent()

            if number_of_points >= 3:
                start, end = sorted(
                    random_generator.sample(
                        range(1, number_of_points), 2
                    )
                )

                child = [None] * number_of_points
                child[0] = 0
                child[start:end + 1] = parent1[start:end + 1]

                remaining = [
                    point
                    for point in parent2
                    if point not in child
                ]

                remaining_index = 0

                for index in range(1, number_of_points):
                    if child[index] is None:
                        child[index] = remaining[remaining_index]
                        remaining_index += 1
            else:
                child = parent1[:]

            if (
                number_of_points > 2
                and random_generator.random() < mutation_rate
            ):
                first_index, second_index = random_generator.sample(
                    range(1, number_of_points), 2
                )
                child[first_index], child[second_index] = (
                    child[second_index],
                    child[first_index],
                )

            new_population.append(child)

        population = new_population

    # Check final population as well.
    for route in population:
        distance = route_distance(route)

        if distance < best_distance:
            best_route = route[:]
            best_distance = distance

    return best_route, float(best_distance)


def compare_routes(coordinates, optimized_route):
    """
    Compare the original collection order with the optimized route.

    Both distances use straight-line GPS distances and return
    to the starting point. The result is a demonstration metric.
    """
    if not coordinates:
        raise ValueError("Add at least one collection point.")

    original_route = list(range(len(coordinates)))
    optimized_route = list(optimized_route)

    expected_points = set(range(len(coordinates)))

    if len(optimized_route) != len(coordinates):
        raise ValueError(
            "The optimized route must include every collection point."
        )

    if set(optimized_route) != expected_points:
        raise ValueError(
            "The optimized route must contain each point index exactly once."
        )

    original_distance = calculate_route_distance(
        original_route, coordinates
    )

    optimized_distance = calculate_route_distance(
        optimized_route, coordinates
    )

    distance_saved = original_distance - optimized_distance

    improvement_percent = (
        (distance_saved / original_distance) * 100
        if original_distance > 0
        else 0.0
    )

    return {
        "original_route": original_route,
        "optimized_route": optimized_route,
        "original_distance_km": float(original_distance),
        "optimized_distance_km": float(optimized_distance),
        "distance_saved_km": float(distance_saved),
        "improvement_percent": float(improvement_percent),
    }

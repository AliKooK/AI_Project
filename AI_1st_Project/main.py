# ======================== IMPORT LIBRARIES ========================
import math  # For mathematical operations (e.g., sqrt, hypot)
import random  # For generating random numbers and selections
import copy  # For deep copying objects
import time  # For tracking execution time
import matplotlib.pyplot as plt  # For plotting


# ======================== COMMON HELPER FUNCTIONS ========================
def get_positive_integer(prompt):
    """Prompt user for a positive integer input with validation."""
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def get_positive_float(prompt):
    """Prompt user for a positive float input with validation."""
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def plot_routes(routes, unassigned, shop_location=(0, 0)):
    """Plot vehicle delivery routes, shop location, and unassigned packages."""
    plt.figure(figsize=(10, 10))
    plt.scatter(shop_location[0], shop_location[1], c='red', marker='s', label='Shop')

    for i, route in enumerate(routes):
        if route:
            x_coords = [shop_location[0]] + [point[0] for point in route] + [shop_location[0]]
            y_coords = [shop_location[1]] + [point[1] for point in route] + [shop_location[1]]
            plt.plot(x_coords, y_coords, label=f'Vehicle {i + 1}')
            for point in route:
                plt.scatter(point[0], point[1], c='blue')

    if unassigned:
        unassigned_x = [point[0] for point in unassigned]
        unassigned_y = [point[1] for point in unassigned]
        plt.scatter(unassigned_x, unassigned_y, c='green', marker='x', label='Unassigned')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Vehicle Delivery Routes')
    plt.legend()
    plt.grid(True)
    plt.show()


# ======================== SIMULATED ANNEALING COMPONENTS ========================
class SAPackage:
    """Represents a package with destination coordinates, weight, and priority."""

    def __init__(self, destination, weight, priority):
        self.destination = destination  # Tuple (x, y) coordinates
        self.weight = weight  # Package weight in kg
        self.priority = priority  # Priority level (1=highest, 5=lowest)


class SAVehicle:
    """Represents a delivery vehicle with capacity and package assignments."""

    def __init__(self, capacity):
        self.capacity = capacity  # Maximum weight capacity
        self.assigned_packages = []  # List of assigned SAPackage objects

    def add_package(self, pkg):
        """Attempt to add a package. Returns True if successful, False if capacity exceeded."""
        if self.total_weight() + pkg.weight > self.capacity:
            return False
        self.assigned_packages.append(pkg)
        return True

    def total_weight(self):
        """Calculate total weight of assigned packages."""
        return sum(pkg.weight for pkg in self.assigned_packages)


def sa_euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points (tuples)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def sa_vehicle_cost(vehicle, penalty_constant=10):
    """
    Calculate cost for a vehicle's route including:
    - Total travel distance
    - Priority-based penalties for delivery order violations
    """
    shop_location = (0, 0)
    total_distance = 0.0
    current_loc = shop_location
    penalty = 0  # Penalty for priority order violations

    # Calculate route distance and priority penalties
    for idx, pkg in enumerate(vehicle.assigned_packages):
        total_distance += sa_euclidean_distance(current_loc, pkg.destination)
        current_loc = pkg.destination
        # Check if current package has lower priority than previous
        if idx > 0:
            prev_pkg = vehicle.assigned_packages[idx - 1]
            if pkg.priority < prev_pkg.priority:
                penalty += penalty_constant * (prev_pkg.priority - pkg.priority)

    # Add return distance to shop
    total_distance += sa_euclidean_distance(current_loc, shop_location)
    return total_distance + penalty


def sa_total_cost(vehicles, penalty_constant=10):
    """Calculate total cost for all vehicles including unassigned package penalties."""
    return sum(sa_vehicle_cost(v, penalty_constant) for v in vehicles)


def sa_initial_solution(packages, num_vehicles, vehicle_capacity):
    """
    Generate initial solution by:
    1. Sorting packages by priority (highest first)
    2. Greedily assigning to first available vehicle
    Returns tuple: (list of vehicles, list of unassigned packages)
    """
    vehicles = [SAVehicle(vehicle_capacity) for _ in range(num_vehicles)]
    unassigned_packages = []
    sorted_pkgs = sorted(packages, key=lambda x: x.priority)  # Sort by priority

    for pkg in sorted_pkgs:
        assigned = False
        # Try to assign to first available vehicle
        for v in vehicles:
            if v.add_package(pkg):
                assigned = True
                break
        if not assigned:
            unassigned_packages.append(pkg)

    return vehicles, unassigned_packages


def sa_neighbor_solution(current_solution, unassigned_packages):
    """
    Generate neighboring solution through one of three mutations:
    1. Add random unassigned package (40% chance)
    2. Move package between vehicles (40% chance)
    3. Swap package order in same vehicle (20% chance)
    """
    new_solution = copy.deepcopy(current_solution)
    new_unassigned = copy.deepcopy(unassigned_packages)
    r = random.random()  # Determine mutation type

    # Case 1: Add unassigned package
    if r < 0.4 and new_unassigned:
        vehicle = random.choice(new_solution)
        pkg = random.choice(new_unassigned)
        if vehicle.add_package(pkg):
            new_unassigned.remove(pkg)

    # Case 2: Move package between vehicles
    elif r < 0.8:
        if len(new_solution) < 2:
            return new_solution, new_unassigned
        vehicle_from, vehicle_to = random.sample(new_solution, 2)
        if not vehicle_from.assigned_packages:
            return new_solution, new_unassigned
        pkg = random.choice(vehicle_from.assigned_packages)
        if vehicle_to.add_package(pkg):
            vehicle_from.assigned_packages.remove(pkg)

    # Case 3: Swap package order in vehicle
    else:
        vehicle = random.choice(new_solution)
        if len(vehicle.assigned_packages) >= 2:
            idx1, idx2 = random.sample(range(len(vehicle.assigned_packages)), 2)
            vehicle.assigned_packages[idx1], vehicle.assigned_packages[idx2] = (
                vehicle.assigned_packages[idx2], vehicle.assigned_packages[idx1]
            )

    return new_solution, new_unassigned


def simulated_annealing(packages, num_vehicles, vehicle_capacity,
                        initial_temp=1000, cooling_rate=0.95,
                        stopping_temp=1, iterations_per_temp=100, penalty_constant=10):
    """
    Simulated Annealing optimization process:
    1. Start with initial solution
    2. Gradually cool temperature while exploring neighboring solutions
    3. Accept worse solutions probabilistically to escape local optima
    Returns optimized solution with metrics
    """
    start_time = time.time()
    # Initialize solution
    vehicles, unassigned_packages = sa_initial_solution(packages, num_vehicles, vehicle_capacity)
    current_solution = vehicles
    current_unassigned = unassigned_packages
    # Calculate initial cost (distance + penalties)
    current_cost = sa_total_cost(current_solution, penalty_constant) + len(current_unassigned) * 1000
    best_solution = copy.deepcopy(current_solution)
    best_unassigned = copy.deepcopy(current_unassigned)
    best_cost = current_cost
    temp = initial_temp  # Current temperature

    # Annealing loop
    while temp > stopping_temp:
        for _ in range(iterations_per_temp):
            # Generate and evaluate neighbor solution
            new_solution, new_unassigned = sa_neighbor_solution(current_solution, current_unassigned)
            new_cost = sa_total_cost(new_solution, penalty_constant) + len(new_unassigned) * 1000
            cost_diff = new_cost - current_cost

            # Acceptance criteria
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
                current_solution = new_solution
                current_unassigned = new_unassigned
                current_cost = new_cost
                # Update best solution if improved
                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_unassigned = copy.deepcopy(current_unassigned)
                    best_cost = current_cost

        # Cooling phase
        print(f"Temperature: {temp:.2f}, Current Cost: {current_cost:.3f}, Best Cost: {best_cost:.3f}")
        temp *= cooling_rate

    end_time = time.time()
    execution_time = end_time - start_time
    return best_solution, best_unassigned, best_cost, execution_time


def sa_print_solution(vehicles, unassigned_packages, best_cost, execution_time):
    """Display optimized solution details for Simulated Annealing."""
    print(f"\nBest total cost (distance + priority penalties + unassigned penalties): {best_cost:.3f}")
    print(f"Execution time: {execution_time:.2f} seconds")

    # Capacity utilization summary
    total_capacity_used = sum(v.total_weight() for v in vehicles)
    total_vehicle_capacity = sum(v.capacity for v in vehicles)
    print(f"\nTotal Capacity Used: {total_capacity_used:.1f} kg / {total_vehicle_capacity:.1f} kg")

    # Vehicle details
    for i, vehicle in enumerate(vehicles):
        status = "OK"
        print(f"\nVehicle {i + 1} ({status}): {vehicle.total_weight():.1f}/{vehicle.capacity} kg")
        for pkg in vehicle.assigned_packages:
            print(f"  - Destination: {pkg.destination}, Weight: {pkg.weight} kg, Priority: {pkg.priority}")

    # Unassigned packages
    if unassigned_packages:
        print("\n UNASSIGNED PACKAGES DUE TO CAPACITY:")
        for pkg in unassigned_packages:
            print(f"  - Destination: {pkg.destination}, Weight: {pkg.weight} kg (Too heavy for any vehicle)")

    # Plot the routes
    sa_routes = []
    for vehicle in vehicles:
        route = [pkg.destination for pkg in vehicle.assigned_packages]
        sa_routes.append(route)
    unassigned_points = [pkg.destination for pkg in unassigned_packages]
    plot_routes(sa_routes, unassigned_points)


# ======================== GENETIC ALGORITHM COMPONENTS ========================
class GAPackage:
    """Represents a package with additional GA-specific metadata."""

    def __init__(self, PID, destination, weight, priority):
        self.PID = PID  # Package ID
        self.Destination = destination  # Dict with 'x' and 'y' coordinates
        self.Weight = weight  # Package weight in kg
        self.Priority = priority  # Priority level (1=highest, 5=lowest)
        self.VID = -1  # Assigned vehicle ID
        self.Efficiency = 100.0  # Delivery efficiency percentage

    def clone(self):
        """Create a deep copy of the package."""
        return copy.deepcopy(self)


class GAVehicle:
    """Represents a vehicle with capacity and route information."""

    def __init__(self, VID, capacity):
        self.VID = VID  # Vehicle ID
        self.capacity = capacity  # Maximum capacity
        self.available_capacity = capacity  # Remaining capacity
        self.packages = []  # Assigned packages
        self.route_distance = 0.0  # Total route distance

    def clone(self):
        """Create a deep copy of the vehicle."""
        return copy.deepcopy(self)


def ga_calculate_route_distance(packages):
    """Calculate total route distance including return to origin."""
    current = {"x": 0, "y": 0}
    distance = 0.0
    for pkg in packages:
        dx = pkg.Destination["x"] - current["x"]
        dy = pkg.Destination["y"] - current["y"]
        distance += math.hypot(dx, dy)
        current = pkg.Destination
    # Add return distance to shop
    dx = 0 - current["x"]
    dy = 0 - current["y"]
    return distance + math.hypot(dx, dy)


class GAChromosome:
    """Represents a potential solution in the genetic algorithm population."""

    def __init__(self, vehicles, packages):
        self.vehicles = [v.clone() for v in vehicles]  # List of GAVehicle
        self.packages = [p.clone() for p in packages]  # List of GAPackage
        self.total_distance = 0.0  # Total distance across all vehicles
        self.avg_efficiency = 0.0  # Average package efficiency
        self.fitness = 0.0  # Overall solution quality score
        self.initialize()

    def initialize(self):
        """Initialize package assignments using a greedy approach."""
        for pkg in self.packages:
            random.shuffle(self.vehicles)
            for vehicle in self.vehicles:
                if vehicle.available_capacity >= pkg.Weight:
                    vehicle.packages.append(pkg)
                    vehicle.available_capacity -= pkg.Weight
                    pkg.VID = vehicle.VID
                    break
        self.evaluate()

    def evaluate(self):
        """Calculate fitness based on distance and efficiency metrics."""
        total_distance = 0.0
        total_efficiency = 0.0

        for vehicle in self.vehicles:
            if vehicle.packages:
                # Calculate route distance
                distance = ga_calculate_route_distance(vehicle.packages)
                vehicle.route_distance = distance
                total_distance += distance

                # Calculate package efficiencies
                for pkg in vehicle.packages:
                    direct_distance = math.hypot(
                        pkg.Destination["x"] - 0,
                        pkg.Destination["y"] - 0
                    )
                    detour = vehicle.route_distance - direct_distance
                    priority_weight = 6 - pkg.Priority  # Higher priority = more weight
                    pkg.Efficiency = max(0, 100 - priority_weight * (detour / 10))
                    total_efficiency += pkg.Efficiency

        self.total_distance = total_distance
        self.avg_efficiency = total_efficiency / len(self.packages) if self.packages else 0
        # Fitness combines inverse distance (70%) and efficiency (30%)
        self.fitness = (0.7 * (1 / (1 + self.total_distance))) + (0.3 * (self.avg_efficiency / 100))

    def clone(self):
        """Create a deep copy of the chromosome."""
        return copy.deepcopy(self)


def ga_crossover(parent1, parent2):
    """
    Perform crossover between two parent chromosomes.
    Exchanges package assignments between vehicles to create offspring.
    """
    child1 = parent1.clone()
    child2 = parent2.clone()

    for p1, p2 in zip(child1.packages, child2.packages):
        if random.random() < 0.5:
            vid1 = p1.VID
            vid2 = p2.VID

            # Update child1
            for v in child1.vehicles:
                if v.VID == vid1:
                    v.packages = [p for p in v.packages if p.PID != p1.PID]
                    v.available_capacity += p1.Weight
                if v.VID == vid2 and v.available_capacity >= p1.Weight:
                    v.packages.append(p1)
                    v.available_capacity -= p1.Weight
                    p1.VID = vid2

            # Update child2
            for v in child2.vehicles:
                if v.VID == vid2:
                    v.packages = [p for p in v.packages if p.PID != p2.PID]
                    v.available_capacity += p2.Weight
                if v.VID == vid1 and v.available_capacity >= p2.Weight:
                    v.packages.append(p2)
                    v.available_capacity -= p2.Weight
                    p2.VID = vid1

    child1.evaluate()
    child2.evaluate()
    return child1, child2


def ga_mutate(chromosome):
    """Perform mutation by either moving a package or swapping delivery order."""
    vehicle = random.choice(chromosome.vehicles)
    if not vehicle.packages:
        return

    pkg = random.choice(vehicle.packages)
    candidate_vehicles = [v for v in chromosome.vehicles if v.VID != vehicle.VID]

    # Attempt to move package to another vehicle
    if candidate_vehicles:
        new_vehicle = random.choice(candidate_vehicles)
        if new_vehicle.available_capacity >= pkg.Weight:
            vehicle.packages.remove(pkg)
            vehicle.available_capacity += pkg.Weight
            new_vehicle.packages.append(pkg)
            new_vehicle.available_capacity -= pkg.Weight
            pkg.VID = new_vehicle.VID
            chromosome.evaluate()
    # If no other vehicles, swap package order
    else:
        if len(vehicle.packages) > 1:
            idx1, idx2 = random.sample(range(len(vehicle.packages)), 2)
            vehicle.packages[idx1], vehicle.packages[idx2] = vehicle.packages[idx2], vehicle.packages[idx1]
            chromosome.evaluate()


def genetic_algorithm(packages, num_vehicles, vehicle_capacity,
                      mutation_rate=0.05, population_size=50,
                      num_generations=500, crossover_rate=0.95):
    """
    Genetic Algorithm optimization process:
    1. Initialize population of random solutions
    2. Evolve population through selection, crossover, and mutation
    3. Return best solution after specified generations
    """
    base_vehicles = [GAVehicle(VID, vehicle_capacity) for VID in range(num_vehicles)]
    population = [GAChromosome(base_vehicles, packages) for _ in range(population_size)]
    best_solution = max(population, key=lambda c: c.fitness)

    # Evolution loop
    for generation in range(num_generations):
        population.sort(key=lambda c: -c.fitness)
        elite = population[:population_size // 10]  # Keep top 10% as elite
        children = []

        # Generate offspring
        while len(children) < population_size - len(elite):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            if random.random() < crossover_rate:
                child1, child2 = ga_crossover(parent1, parent2)
                children.extend([child1, child2])
            else:
                children.extend([parent1.clone(), parent2.clone()])

        # Apply mutations
        for child in children:
            if random.random() < mutation_rate:
                ga_mutate(child)

        # Create new population
        population = elite + children[:population_size - len(elite)]
        current_best = max(population, key=lambda c: c.fitness)
        if current_best.fitness > best_solution.fitness:
            best_solution = current_best.clone()

        # Progress reporting
        if generation % 50 == 0:
            print(f"Gen {generation}: Dist={best_solution.total_distance:.2f} "
                  f"Eff={best_solution.avg_efficiency:.2f}% "
                  f"Fitness={best_solution.fitness:.4f}")

    return best_solution


# ======================== MENU SYSTEM ========================
def run_annealing():
    """Execute Simulated Annealing optimization with user inputs."""
    print("\n=== Package Delivery Optimization using Simulated Annealing ===")

    # Get user inputs
    num_vehicles = get_positive_integer("Enter number of available vehicles: ")
    capacity = get_positive_float("Enter maximum capacity per vehicle (kg): ")
    num_packages = get_positive_integer("Enter number of packages: ")

    # Collect package data
    packages = []
    for i in range(num_packages):
        print(f"\nData for package {i + 1}:")
        x = get_positive_float("  Destination X coordinate (0-100 km): ")
        y = get_positive_float("  Destination Y coordinate (0-100 km): ")
        weight = get_positive_float("  Package weight (kg): ")
        priority = get_positive_integer("  Package priority (1=highest, 5=lowest): ")
        if weight > capacity:
            print(
                f"Warning: Package {i + 1} weight ({weight} kg) exceeds vehicle capacity ({capacity} kg). It will be unassigned.")
        packages.append(SAPackage((x, y), weight, priority))

    # Run optimization
    best_solution, best_unassigned, best_cost, execution_time = simulated_annealing(
        packages, num_vehicles, capacity
    )

    # Display results
    sa_print_solution(best_solution, best_unassigned, best_cost, execution_time)
    input("\nPress Enter to return to main menu...")


def run_genetic():
    """Execute Genetic Algorithm optimization with user inputs."""
    print("\n=== Package Delivery Optimization using Genetic Algorithm ===")

    # Get user inputs
    num_vehicles = get_positive_integer("Enter number of available vehicles: ")
    capacity = get_positive_float("Enter maximum capacity per vehicle (kg): ")
    num_packages = get_positive_integer("Enter number of packages: ")

    # Collect package data
    packages = []
    for i in range(num_packages):
        print(f"\nData for package {i + 1}:")
        x = get_positive_float("  Destination X coordinate (0-100 km): ")
        y = get_positive_float("  Destination Y coordinate (0-100 km): ")
        weight = get_positive_float("  Package weight (kg): ")
        priority = get_positive_integer("  Package priority (1=highest, 5=lowest): ")
        packages.append(GAPackage(i, {"x": x, "y": y}, weight, priority))

    # Run optimization
    best = genetic_algorithm(
        packages, num_vehicles, capacity,
        mutation_rate=0.05,
        population_size=100,
        num_generations=500,
        crossover_rate=0.95
    )

    # Display results
    print("\n=== Best Solution ===")
    print(f"Total Distance: {best.total_distance:.2f} km")
    print(f"Average Efficiency: {best.avg_efficiency:.2f}%")
    print(f"Fitness Score: {best.fitness:.4f}")

    # Calculate utilization metrics
    total_weight = 0
    total_capacity = sum(v.capacity for v in best.vehicles)
    unassigned = []

    print("\nVehicle Assignments:")
    for vehicle in best.vehicles:
        vehicle_weight = sum(p.Weight for p in vehicle.packages)
        status = "OK" if vehicle_weight <= vehicle.capacity else "OVERLOADED"
        print(f"\nVehicle {vehicle.VID} ({status}): {vehicle_weight:.1f}/{vehicle.capacity} kg")
        total_weight += vehicle_weight
        if vehicle.packages:
            print(f"  Route distance: {vehicle.route_distance:.2f} km")
            pkg_ids = [p.PID for p in vehicle.packages]
            print(f"  Packages: {pkg_ids}")

    # Identify unassigned packages
    all_assigned = [p.PID for v in best.vehicles for p in v.packages]
    unassigned = [p for p in best.packages if p.PID not in all_assigned]

    if unassigned:
        print("\n UNASSIGNED PACKAGES DUE TO CAPACITY:")
        for pkg in unassigned:
            print(f"  Package {pkg.PID}: Weight {pkg.Weight} kg (Too heavy for any vehicle)")

    print(f"\nTotal Capacity Utilization: {total_weight:.1f}/{total_capacity:.1f} kg")

    # Display package efficiencies
    print("\nPackage Efficiencies:")
    for pkg in sorted(best.packages, key=lambda p: p.PID):
        print(f"Package {pkg.PID}: "
              f"Vehicle {pkg.VID} | "
              f"Eff={pkg.Efficiency:.1f}% | "
              f"Priority={pkg.Priority} (1=highest)")

    # Plot the routes
    ga_routes = []
    for vehicle in best.vehicles:
        route = [(pkg.Destination['x'], pkg.Destination['y']) for pkg in vehicle.packages]
        ga_routes.append(route)
    unassigned_points = [(pkg.Destination['x'], pkg.Destination['y']) for pkg in unassigned]
    plot_routes(ga_routes, unassigned_points)

    input("\nPress Enter to return to main menu...")


def main_menu():
    """Main menu system for user interaction."""
    print("\nWelcome to Package Delivery Route Optimizer!")
    while True:
        print("\n" + "=" * 40)
        print("MAIN MENU")
        print("=" * 40)
        print("1. Genetic Algorithm Optimization")
        print("2. Simulated Annealing Optimization")
        print("3. Exit Program")
        print("=" * 40)

        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            run_genetic()
        elif choice == '2':
            run_annealing()
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
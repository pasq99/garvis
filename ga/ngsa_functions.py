import random
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from deap import base, creator, tools, algorithms
from deap.tools.emo import assignCrowdingDist
from myutils.geometry2d import generate_random_point, is_valid_path, distance, compute_curvature
import time
from scenario.conf import scn


class Individual():
    def __init__(self, av):
        # an av is defined by a start point, an end point, 
        # a collision point and a speed
        self.entry = av[0]
        self.exit = av[1]
        self.collision_point = av[2]
        self.speed = av[3]  # in NM/min
        self.genes = None
        self.route = None

    def individual_from_seed (ind):
        new_ind = Individual((ind.entry,ind.exit,None,ind.speed))
        new_ind.genes = ind.genes.copy()
        new_ind.route = None 
        # modify randomly one gene
        i = random.randint(1, len(new_ind.genes) - 1)
        new_ind.genes[i] = {
            'time': random.uniform(new_ind.genes[i-1]['time'], new_ind.genes[i+1]['time']),
            'angle': random.uniform(-scn['max_turn_angle_deg']/2, -scn['max_turn_angle_deg']/2),
            'speed': random.uniform(speed_min, speed_max)
        }
        new_ind.route = new_ind._route()
        return new_ind

    def create_genes(self, n_segments, scn):
        self.genes = []
        current_time = 0
        next_time = (distance(self.entry, self.collision_point)-scn['separation_min']) / self.speed  # Tempo in minuti
        for _ in range(n_segments-1):
            # each gene is defined by a time, a turn angle and a speed
            gene = {}
            gene['time'] = random.uniform(current_time, next_time)  # Tempo in minuti
            current_time = gene['time']
            next_time = distance(self.entry, self.exit) / self.speed
            gene['angle'] = random.uniform(-scn['max_turn_angle_deg'], scn['max_turn_angle_deg'])  # Angolo in radianti
            gene['speed'] = random.uniform(speed_min, speed_max)  # Velocità in NM/min
            self.genes.append(gene)
        self.collision_point = None
        self.route = self._route()

    def get_route(self):
        if self.route is None:
            logger.warning(f"Route is None, create genes first.")
        return self.route
    
    def get_path(self):
        return self.route['path']

    def _route(self):
        path = [(self.entry[0], self.entry[1], 0)]
       
        current_point = self.entry
        current_angle = math.atan2(self.exit[1] - self.entry[1], self.exit[0] - self.entry[0])
        current_speed = self.speed
        for gene in self.genes:
            distance_travelled = current_speed* (gene['time'] - path[-1][2])  # Distanza in NM
            new_x = current_point[0] + distance_travelled * math.cos(current_angle)
            new_y = current_point[1] + distance_travelled * math.sin(current_angle)
            current_angle += math.radians(gene['angle'])
            current_point = (new_x, new_y, gene['time'])
            current, scn, see_speed = gene['speed']
            path.append(current_point)
        path.append((self.exit[0], self.exit[1], distance(self.entry, self.exit) / self.speed))
        return {'path': [(p[0], p[1]) for p in path], 'times': [p[2] for p in path] }



def mutate_individual(ind, scn, indpb=0.3):
    """Applica una mutazione sui punti intermedi."""
    current_time = 0
    for i in range(0, len(ind.genes) - 1):  # Non muovere entry e exit
            if random.random() < indpb:
                ind.genes[i] = {
                    'time': random.uniform(current_time, ind.genes[i+1]['time']),
                    'angle': random.uniform(-scn['max_turn_angle_deg']/2, -scn['max_turn_angle_deg']/2),
                    'speed': random.uniform(speed_min, speed_max)
                }
            current_time = ind.genes[i]['time']
            ind.route = ind._route()
    return (ind,)

'''
Da rivedere
'''
def mutate_swap_segments(ind, prob=0.3):
    """Scambia due punti intermedi della traiettoria con una certa probabilità."""
    if random.random() < prob and len(ind) > 3:
        i, j = sorted(random.sample(range(1, len(ind) - 1), 2))
        ind[i], ind[j] = ind[j], ind[i]
    return (ind,)


def random_population(pop_size, scn, n_segments, av):
    population = []
    while len(population) < pop_size:
        ind = Individual(av)
        ind.create_genes(n_segments, scn)
        if is_valid_path(ind.get_path(), scn):
            population.append(ind)
    return population


def population_from_seed(pop_size, scn, seed=None):
    if seed is None:
        logger.error("Seed is None, cannot create population.")
        return None

    population = []
    '''
    To be fixed: should use the Individual object
    '''
    for _ in range(pop_size):
        mutant = creator.Individual(perturb_seed(seed, scn))
        mutant, = combined_mutation(mutant)
        if is_valid_path(mutant, scn):
            population.append(mutant)
    return population

# fitness function
def evaluate(ind, original_exit, original_line):
    path = LineString(ind.get_path())

    # Calcola distanza finale
    goal_dist_raw = distance(path[-1], original_exit)

    # Penalità solo se troppo lontano
    max_allowed_dist = 5.0  # ad esempio, 2 NM
    goal_dist = max(0, goal_dist_raw - max_allowed_dist)

    # Obiettivo 2: fluidità della traiettoria
    curvature = compute_curvature(path)

    # Obiettivo 3: deviazione dalla rotta originale
    avg_dev = np.mean([original_line.distance(Point(x, y)) for x, y in path.coords])

    # Rumore per spezzare simmetrie
    noise = random.uniform(-0.05, 0.05)

    return (goal_dist, -curvature + noise, -avg_dev)

def crossover_individuals(ind1, ind2):
    """Prende un istante tra l'inizio e la fine e separa lì"""
    min_exit_time = min(ind1.genes[-1]['time'], ind2.genes[-1]['time'])
    cross_time = random.uniform(ind1.genes[0]['time'], min_exit_time)
    new1_genes = [gene for gene in ind1.genes if gene['time'] <= cross_time] + \
                 [gene for gene in ind2.genes if gene['time'] > cross_time]
    new2_genes = [gene for gene in ind2.genes if gene['time'] <= cross_time] + \
                 [gene for gene in ind1.genes if gene['time'] > cross_time]
    new1 = Individual((ind1.entry, ind1.exit, None, ind1.speed))
    new2 = Individual((ind2.entry, ind2.exit, None, ind2.speed))
    new1.genes = new1_genes
    new2.genes = new2_genes 
    new1.route = new1._route()
    new2.route = new2._route()
    return new1, new2

def combined_mutation(ind):
    ind, = mutate_individual(ind, scn, indpb=0.9)
    ind, = mutate_swap_segments(ind, prob=0.5)
    return (ind,)

def create_individual(av, scn, n_segments=3):
    ind = Individual(av)
    ind.create_genes(n_segments, scn)
    return creator.Individual(ind)


def perturb_seed(seed,scn, intensity=15):
    """Applica una variazione casuale ai punti intermedi della seed."""
    perturbed = seed[:]
    for i in range(1, len(seed) - 1):
        x_off = random.uniform(-intensity, intensity)
        y_off = random.uniform(-intensity, intensity)
        perturbed[i] = (
            min(max(perturbed[i][0] + x_off, 0), scn["area_size"]),
            min(max(perturbed[i][1] + y_off, 0), scn["area_size"]),
        )
    return perturbed
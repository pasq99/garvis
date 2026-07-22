import matplotlib.pyplot as plt
import json
import logging
from shapely.geometry import LineString, Polygon
import glob
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
})


filnames = glob.glob("data/*_pareto_front_*.json")

for filename in filnames:
    results = json.load(open(filename))
    plt.figure(figsize=(10, 7))
    population = results['population']
    scenario = results['scenario']

    # entry time and exit time of erly plan in collision point
    collision_time = scenario["collision_time"]
    time_slot = [collision_time - 60*2.5/scenario["speed_early"], collision_time + 60*2.5/scenario["speed_early"]]


    for individual in population:
        x = [p[0] for p in individual['path']]
        y = [p[1] for p in individual['path']]
        plt.plot(x, y, color='b', alpha=0.3)
        # highlight the part of the path in the time slot
        path = LineString(individual['path'])
        points_in_slot = [pt for pt in individual['path'] if time_slot[0] <= pt[2] <= time_slot[1]]
        if len(points_in_slot) >= 2:
            path_in_slot = LineString(points_in_slot[:2])
            if path_in_slot.length > 0:
                plt.plot(*path_in_slot.xy, color='r', linewidth=2)

    # plot a circle
    circle = plt.Circle(scenario["collision_point"], 2.5, color='r', fill=True)
    plt.gca().add_artist(circle)
    plt.plot([scenario["ing_early"][0], scenario["usc_early"][0]],[scenario["ing_early"][1], scenario["usc_early"][1]], color='g', label='Early UAV')
    plt.plot([scenario["ing_late"][0], scenario["usc_late"][0]],[scenario["ing_late"][1], scenario["usc_late"][1]], color='orange', label='Late UAV')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.xlabel("X (NM)")
    plt.ylabel("Y (NM)")
    plt.title("Pareto Front Trajectories")
    image_name = filename.replace(".json", ".png")
    plt.savefig(image_name)

    # draw a 3d scatter plot of the fitness values
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111, projection='3d')
    for individual in population:
        ax.scatter(individual['fitness'][0], individual['fitness'][1], individual['fitness'][2], alpha=0.5)
    ax.set_xlabel("Exit Distance")
    ax.set_ylabel("Path Length")
    ax.set_zlabel("Orientation Change on Exit")
    #plt.title("Pareto Front Fitness Values")
    plt.savefig(image_name.replace(".png", "_fitness.png"))

    # draw 3 scatter plots of the fitness values
    fig, axese = plt.subplots(3,1, figsize=(10, 15))
    for individual in population:
        axese[0].scatter(individual['fitness'][1], individual['fitness'][2], alpha=0.5)
        axese[1].scatter(individual['fitness'][0], individual['fitness'][2], alpha=0.5)
        axese[2].scatter(individual['fitness'][0], individual['fitness'][1], alpha=0.5)
    axese[0].set_xlabel(" exit distance (NM)")
    axese[0].set_ylabel(" fluidity")
    axese[1].set_xlabel(" path_length (NM)")
    axese[1].set_ylabel(" fluidity")
    axese[2].set_xlabel(" path_length (NM)")
    axese[2].set_ylabel(" exit distance (NM)")
    #plt.title("Pareto Front Fitness Values")
    plt.savefig(image_name.replace(".png", "_fitness_2d.png"))
    plt.close('all')
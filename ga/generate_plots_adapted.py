import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import logging
import glob
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

def quota_ft_to_nm(z_ft):
    # Converte i piedi (ft) in miglia nautiche (NM)
    return z_ft / 6076.12

# Carica i file del fronte di Pareto
filenames = glob.glob("data/*_pareto_front_*.json")

if not filenames:
    logger.warning("Nessun file JSON trovato in data/")

for filename in filenames:
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Errore lettura {filename}: {e}")
        continue
        
    if "population" not in results or "scenario" not in results:
        continue
        
    population = results['population']
    scenario = results['scenario']
    
    if not population:
        continue

    # Compatibilità con JSON vecchi
    if "ac0" not in scenario or "ac1" not in scenario:
        logger.warning(f"File {filename} usa un vecchio formato. Skippato.")
        continue

    # Estrazione coordinate Scenario (nuova struttura 4D)
    p0_ac0 = scenario["ac0"]["p0"]
    pt_ac0 = scenario["ac0"]["p_target"]
    p0_ac1 = scenario["ac1"]["p0"]
    pt_ac1 = scenario["ac1"]["p_target"]
    
    image_name_base = filename.replace(".json", "")

    # ==========================================
    # 1. GRAFICO TRAIETTORIE 2D (Planimetria)
    # ==========================================
    plt.figure(figsize=(10, 8))
    for idx, individual in enumerate(population):
        if "path" in individual and individual["path"]:
            x = [p[0] for p in individual['path']]
            y = [p[1] for p in individual['path']]
            alpha_val = 1.0 if idx == 0 else 0.3
            color_val = 'r' if idx == 0 else 'b'
            plt.plot(x, y, color=color_val, alpha=alpha_val, linewidth=2 if idx == 0 else 1)
            
    # Marker AC0 (Risolutore)
    plt.scatter(p0_ac0[0], p0_ac0[1], color='blue', marker='o', s=100, label='Start AC0', zorder=5)
    plt.scatter(pt_ac0[0], pt_ac0[1], color='darkgreen', marker='X', s=120, label='Target AC0', zorder=5)
    
    # Marker AC1 (Intruso)
    plt.plot([p0_ac1[0], pt_ac1[0]], [p0_ac1[1], pt_ac1[1]], color='orange', linestyle='--', linewidth=2, label='Path AC1', zorder=4)
    plt.scatter(p0_ac1[0], p0_ac1[1], color='orange', marker='s', s=80, label='Start AC1', zorder=5)
    plt.scatter(pt_ac1[0], pt_ac1[1], color='darkred', marker='P', s=100, label='Target AC1', zorder=5)

    plt.xlabel("X (NM)")
    plt.ylabel("Y (NM)")
    plt.title(f"Pareto Front Trajectories (2D)\n{os.path.basename(filename)}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{image_name_base}_traiettorie_2d.png", bbox_inches='tight')
    plt.close()

    # ==========================================
    # 2. GRAFICO TRAIETTORIE 3D (Quota in FEET)
    # ==========================================
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    for idx, individual in enumerate(population):
        if "path" in individual and individual["path"]:
            x = [p[0] for p in individual['path']]
            y = [p[1] for p in individual['path']]
            # Quota NON CONVERTITA, lasciata in piedi (ft) come richiesto
            z = [p[2] for p in individual['path']]
            alpha_val = 1.0 if idx == 0 else 0.3
            color_val = 'r' if idx == 0 else 'b'
            ax_3d.plot(x, y, z, color=color_val, alpha=alpha_val, linewidth=2 if idx == 0 else 1)

    # AC0 in FEET
    ax_3d.scatter(p0_ac0[0], p0_ac0[1], p0_ac0[2], color='blue', marker='o', s=100, label='Start AC0')
    ax_3d.scatter(pt_ac0[0], pt_ac0[1], pt_ac0[2], color='darkgreen', marker='X', s=120, label='Target AC0')
    
    # AC1 in FEET
    ax_3d.plot([p0_ac1[0], pt_ac1[0]], [p0_ac1[1], pt_ac1[1]], [p0_ac1[2], pt_ac1[2]], 
               color='orange', linestyle='--', linewidth=2, label='Path AC1')
    ax_3d.scatter(p0_ac1[0], p0_ac1[1], p0_ac1[2], color='orange', marker='s', s=80, label='Start AC1')
    ax_3d.scatter(pt_ac1[0], pt_ac1[1], pt_ac1[2], color='darkred', marker='P', s=100, label='Target AC1')

    ax_3d.set_xlabel("X (NM)")
    ax_3d.set_ylabel("Y (NM)")
    ax_3d.set_zlabel("Quota (ft)")
    ax_3d.set_title(f"Pareto Front Trajectories (3D)\n{os.path.basename(filename)}")
    ax_3d.legend()
    plt.savefig(f"{image_name_base}_traiettorie_3d.png", bbox_inches='tight')
    plt.close()

    # ==========================================
    # RACCOLTA FITNESS
    # ==========================================
    f1_list, f2_list, f3_list = [], [], []
    for individual in population:
        fit = individual.get('fitness')
        if fit and len(fit) >= 3:
            f1_list.append(fit[0])
            f2_list.append(fit[1])
            f3_list.append(fit[2])
            
    if not f1_list:
        logger.warning(f"Nessun valore di fitness trovato in {filename}")
        continue

    # ==========================================
    # 3. GRAFICO FITNESS 3D
    # ==========================================
    fig_fit = plt.figure(figsize=(10, 8))
    ax_fit = fig_fit.add_subplot(111, projection='3d')
    sc = ax_fit.scatter(f1_list, f2_list, f3_list, c=f3_list, cmap='viridis', s=50, alpha=0.8, edgecolor='k')
    ax_fit.set_xlabel("Obj 1: Distanza Exit (NM)")
    ax_fit.set_ylabel("Obj 2: Lunghezza Path")
    ax_fit.set_zlabel("Obj 3: Smoothness / Penalità")
    ax_fit.set_title(f"Pareto Front Fitness Values (3D)\n{os.path.basename(filename)}")
    fig_fit.colorbar(sc, ax=ax_fit, pad=0.1, shrink=0.6, label="Obj 3: Smoothness")
    plt.savefig(f"{image_name_base}_fitness_3d.png", bbox_inches='tight')
    plt.close()

    # ==========================================
    # 4. GRAFICO FITNESS 2D (SUBPLOTS)
    # ==========================================
    fig_2d, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # f2 vs f3
    axes[0].scatter(f2_list, f3_list, alpha=0.5, color='purple', edgecolor='k')
    axes[0].set_xlabel("Obj 2: Path Length")
    axes[0].set_ylabel("Obj 3: Smoothness / Penalità")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # f1 vs f3
    axes[1].scatter(f1_list, f3_list, alpha=0.5, color='teal', edgecolor='k')
    axes[1].set_xlabel("Obj 1: Exit Distance")
    axes[1].set_ylabel("Obj 3: Smoothness / Penalità")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # f1 vs f2
    axes[2].scatter(f1_list, f2_list, alpha=0.5, color='crimson', edgecolor='k')
    axes[2].set_xlabel("Obj 1: Exit Distance")
    axes[2].set_ylabel("Obj 2: Path Length")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    fig_2d.suptitle(f"Pareto Front Fitness Scatters (2D)\n{os.path.basename(filename)}", y=0.92, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{image_name_base}_fitness_2d.png", bbox_inches='tight')
    plt.close()

    logger.info(f"Salvati 4 grafici per: {os.path.basename(filename)}")

logger.info("Elaborazione completata per tutti i file.")

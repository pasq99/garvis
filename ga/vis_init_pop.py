import os
import json
import logging
import csv
import math
import matplotlib.pyplot as plt
import deap_ga_module_v2 as ga
from scenario.conf import scn

# --- CONFIGURAZIONE LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def carica_scenari_dal_csv_reale(file_path):
    """Legge il CSV reale e accoppia AC0 e AC1 sotto lo stesso ID_Esperimento"""
    scenari = {}
    with open(file_path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            exp_id = int(row["ID_Esperimento"])
            id_aereo = int(row["ID_Aereo"])
            
            if exp_id not in scenari:
                scenari[exp_id] = {}
                
            ac_data = {
                "p0": [float(row["Ingresso_x"]), float(row["Ingresso_y"]), float(row["Ingresso_z"])],
                "prua0": float(row["Prua_Iniziale_deg"]),
                "v0": float(row["Velocita_kmh"]),
                "p_target": [float(row["Uscita_Nominale_x"]), float(row["Uscita_Nominale_y"]), float(row["Uscita_Nominale_z"])],
                "prua_f": float(row["Prua_Finale_deg"]),
                "t_inc": float(row["TempoIncontro_sec"])
            }
            
            if id_aereo == 0:
                scenari[exp_id]["ac0"] = ac_data
            else:
                scenari[exp_id]["ac1"] = ac_data
                
    return scenari

def main():
    logger.info("=== STEP 1: Inizializzazione Popolazione su Dati CSV Reali ===")
    
    # Rilevamento automatico della root del progetto per trovare il CSV
    script_dir = os.path.dirname(os.path.abspath(__file__)) # cartella 'ga'
    project_root = os.path.dirname(script_dir)              # cartella principale del progetto
    
    input_csv = "data/traiettorie_aerei.csv"
        
    logger.info(f"File CSV trovato correttamente in: {input_csv}")
    scenari_db = carica_scenari_dal_csv_reale(input_csv)
    logger.info(f"Caricati con successo {len(scenari_db)} scenari dal CSV.")
    
    base_output_dir = os.path.join(project_root, "data", "initial_population")
    pop_size = scn.get("pop_size", 20)  # Dimensione popolazione da conf
    
    # Cicliamo su tutti gli scenari estratti dal CSV
    for exp_id, exp_scenario in scenari_db.items():
        if "ac0" not in exp_scenario or "ac1" not in exp_scenario:
            continue
            
        logger.info(f"Generazione Gen 0 per lo Scenario ID: {exp_id}")
        exp_dir = os.path.join(base_output_dir, f"exp_{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Calcoliamo dinamicamente il limite dell'area per non tagliare fuori i punti del CSV
        tutti_x = [exp_scenario["ac0"]["p0"][0], exp_scenario["ac0"]["p_target"][0],
                   exp_scenario["ac1"]["p0"][0], exp_scenario["ac1"]["p_target"][0]]
        tutti_y = [exp_scenario["ac0"]["p0"][1], exp_scenario["ac0"]["p_target"][1],
                   exp_scenario["ac1"]["p0"][1], exp_scenario["ac1"]["p_target"][1]]
        
        lato_dinamico = max(max(tutti_x), max(tutti_y)) + 10.0
        
        # 1. Generiamo la popolazione iniziale casuale (Geni del genetico)
        popolazione_iniziale = [ga.crea_individuo() for _ in range(pop_size)]
        
        # 2. Simuliamo la traiettoria dell'intruso AC1 (passando p_target se supportato)
        try:
            sim1 = ga.simula_traiettoria(
                individuo=[], 
                p0=exp_scenario["ac1"]["p0"],
                theta0_deg=exp_scenario["ac1"]["prua0"],
                v0=exp_scenario["ac1"]["v0"],
                t0=0,
                lato=lato_dinamico,
                p_target_scenario=exp_scenario["ac1"]["p_target"]
            )
        except TypeError:
            # Fallback temporaneo se deap_ga_module_v2 non è ancora aggiornato con p_target_scenario
            sim1 = ga.simula_traiettoria(
                individuo=[], 
                p0=exp_scenario["ac1"]["p0"],
                theta0_deg=exp_scenario["ac1"]["prua0"],
                v0=exp_scenario["ac1"]["v0"],
                t0=0,
                lato=lato_dinamico
            )
        
        plt.figure(figsize=(10, 8))
        
        # INIZIALIZZAZIONE MANCANTE CORRETTA QUI:
        pop_data_log = []
        
        # 3. Simuliamo e plottiamo ciascun individuo iniziale dell'aereo controllato AC0
        for idx, ind in enumerate(popolazione_iniziale):
            try:
                sim0 = ga.simula_traiettoria(
                    individuo=ind,
                    p0=exp_scenario["ac0"]["p0"],
                    theta0_deg=exp_scenario["ac0"]["prua0"],
                    v0=exp_scenario["ac0"]["v0"],
                    t0=0,
                    lato=lato_dinamico,
                    p_target_scenario=exp_scenario["ac0"]["p_target"]
                )
            except TypeError:
                sim0 = ga.simula_traiettoria(
                    individuo=ind,
                    p0=exp_scenario["ac0"]["p0"],
                    theta0_deg=exp_scenario["ac0"]["prua0"],
                    v0=exp_scenario["ac0"]["v0"],
                    t0=0,
                    lato=lato_dinamico
                )
            
            path0 = sim0["path"]
            if path0:
                x_ac0 = [p[0] for p in path0]
                y_ac0 = [p[1] for p in path0]
                lbl = "Traiettorie Iniziali AC0 (Gen 0)" if idx == 0 else ""
                plt.plot(x_ac0, y_ac0, color='b', alpha=0.15, linewidth=1, label=lbl)
                
            pop_data_log.append({
                "ind_id": idx,
                "genome": list(ind),
                "path": path0,
                "exit_point": sim0.get("exit_point"),
                "exited": sim0.get("exited")
            })
            
        # 4. Disegno dell'Aereo Intruso AC1
        path1 = sim1["path"]
        if path1:
            x_ac1 = [p[0] for p in path1]
            y_ac1 = [p[1] for p in path1]
            plt.plot(x_ac1, y_ac1, color='r', linestyle='-', linewidth=2, label="Traiettoria AC1 (Intruso)")
            plt.scatter(path1[0][0], path1[0][1], color='r', marker='o', s=80)
            plt.scatter(path1[-1][0], path1[-1][1], color='darkred', marker='x', s=100)
            
        # 5. Marcatori grafici dei punti nominali del CSV (Verdi)
        ing_early = exp_scenario["ac0"]["p0"]
        usc_early = exp_scenario["ac0"]["p_target"]
        plt.scatter(ing_early[0], ing_early[1], color='g', marker='^', s=120, zorder=5, label="Ingresso AC0")
        plt.scatter(usc_early[0], usc_early[1], color='darkgreen', marker='X', s=140, zorder=5, label="Target Uscita AC0")
        
        # Impostazione assi dinamici
        plt.xlim(0, lato_dinamico)
        plt.ylim(0, lato_dinamico)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel("X (NM)")
        plt.ylabel("Y (NM)")
        plt.title(f"Ispezione Gen 0 - Scenario {exp_id} (Scala Reale)")
        plt.legend(loc="upper right")
        
        # Salvataggio file
        plt.savefig(os.path.join(exp_dir, "initial_pop_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(os.path.join(exp_dir, "initial_pop_data.json"), "w") as f:
            json.dump({"exp_id": exp_id, "scenario": exp_scenario, "population": pop_data_log}, f, indent=4)
            
    logger.info(f"Fatto! Controlla la cartella: {os.path.join(project_root, 'data', 'initial_population')}")

if __name__ == "__main__":
    main()
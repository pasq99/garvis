import csv
import json
import random
import math
import argparse
import os
import sys
from collections import Counter
from itertools import product

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fallback in caso non esista il modulo conf
try:
    from scenario.conf import scn
except ImportError:
    scn = {
        "area_size": 100,
        "speed_min": 740,
        "speed_max": 930,
        "t_max_sim": 1200,
        "sampling_interval": 5,
        "separation_min": 2.5,
        "separation_min_vertical": 1000,
        "flight_levels": [31000, 32000, 33000, 34000]
    }

# --- CONFIGURAZIONE AMBIENTE OPERATIVO OTTIMIZZATO ---
AREA_SIZE = scn["area_size"]         # 100 x 100 NM
SPEED_MIN = scn["speed_min"]         # km/h
SPEED_MAX = scn["speed_max"]         # km/h
ROT_MAX_DEG = 3.0                    # 3 gradi/secondo
ROT_MAX_RAD = math.radians(ROT_MAX_DEG)
DT = scn["sampling_interval"]        # Finestra campionamento (secondi)
T_MAX_SIM = scn["t_max_sim"]         # Durata totale (es. 1200s)

# Parametri di separazione minima dello spazio aereo (Bolla ATM 3D generalizzata)
SEPARAZIONE_ORIZZONTALE_MIN = scn["separation_min"]
SEPARAZIONE_VERTICALE_MIN = scn["separation_min_vertical"]

FLIGHT_LEVELS = scn["flight_levels"]
KMH_TO_NM_SEC = 0.539957 / 3600
ANGLE_BINS = ((2, 15), (15, 45), (45, 75), (75, 105), (105, 150), (150, 180))
SPEED_RATIO_BINS = ((0.75, 0.90), (0.95, 1.05), (1.10, 1.40))
TIME_FRACTION_BINS = ((0.20, 0.40), (0.40, 0.60), (0.60, 0.80))
SEVERITY_BINS = {
    "Collision": ((0.0, 0.02), (0.02, 0.06), (0.06, 0.099)),
    "LoS": ((0.10, 0.80), (0.80, 1.70), (1.70, SEPARAZIONE_ORIZZONTALE_MIN - 1e-3)),
    "Safe": ((SEPARAZIONE_ORIZZONTALE_MIN, 4.0), (4.0, 6.0), (6.0, 10.0)),
}

def genera_traiettoria_nominale_4d(x0, y0, z0, prua_deg, velocita_kmh, t_max=T_MAX_SIM):
    path = []
    theta = math.radians(prua_deg)
    v_nm_s = velocita_kmh * KMH_TO_NM_SEC
    
    x, y, z = x0, y0, z0
    for t in range(0, t_max + DT, DT):
        path.append((x, y, z, t))
        x += v_nm_s * math.cos(theta) * DT
        y += v_nm_s * math.sin(theta) * DT
    return path

def calcola_cpa(x0, y0, v0_nm_s, prua0, x1, y1, v1_nm_s, prua1):
    """Calcola matematicamente il Closest Point of Approach (CPA) tra due vettori in moto rettilineo."""
    vx0 = v0_nm_s * math.cos(math.radians(prua0))
    vy0 = v0_nm_s * math.sin(math.radians(prua0))
    vx1 = v1_nm_s * math.cos(math.radians(prua1))
    vy1 = v1_nm_s * math.sin(math.radians(prua1))
    
    dvx = vx0 - vx1
    dvy = vy0 - vy1
    dx = x0 - x1
    dy = y0 - y1
    
    v_rel_sq = dvx**2 + dvy**2
    if v_rel_sq < 1e-6:
        # Volano paralleli alla stessa velocità
        return 0.0, math.sqrt(dx**2 + dy**2)
    
    # t_cpa è il punto di minimo della derivata della distanza rispetto al tempo
    t_cpa = - (dx*dvx + dy*dvy) / v_rel_sq
    t_cpa = max(0.0, t_cpa) # Il CPA non può essere nel passato
    
    dist_h = math.sqrt((dx + dvx*t_cpa)**2 + (dy + dvy*t_cpa)**2)
    return t_cpa, dist_h

def calcola_uscita_bordo(x0, y0, vx, vy, area_size):
    """Calcola il punto e l'istante in cui una traiettoria rettilinea, partendo da
    (x0, y0) con velocita' (vx, vy) [NM/s], esce dal box [0, area_size] x [0, area_size].
    Sostituisce la vecchia 'profezia d'uscita' a tempo fisso T_MAX_SIM, che a queste
    velocita' proietta sempre un punto fuori dall'area (dist percorsa > diagonale del box),
    rendendo impossibile il filtro di contenimento."""
    t_candidates = []
    if vx > 0:
        t_candidates.append((area_size - x0) / vx)
    elif vx < 0:
        t_candidates.append((0.0 - x0) / vx)
    if vy > 0:
        t_candidates.append((area_size - y0) / vy)
    elif vy < 0:
        t_candidates.append((0.0 - y0) / vy)

    t_exit = min(t for t in t_candidates if t > 0)
    x_exit = x0 + vx * t_exit
    y_exit = y0 + vy * t_exit
    # Clamp per sicurezza numerica (arrotondamenti float sul bordo)
    x_exit = min(max(x_exit, 0.0), area_size)
    y_exit = min(max(y_exit, 0.0), area_size)
    return x_exit, y_exit, t_exit

def _balanced_bins(count, size, rng):
    values = [index % size for index in range(count)]
    rng.shuffle(values)
    return values


def _sample_speeds(bin_index, rng):
    ratio = rng.uniform(*SPEED_RATIO_BINS[bin_index])
    lower = max(SPEED_MIN, SPEED_MIN / ratio)
    upper = min(SPEED_MAX, SPEED_MAX / ratio)
    v1 = rng.uniform(lower, upper)
    return v1 * ratio, v1


def calcola_scenari_coerenti(exp_id, spec=None, rng=None, max_attempts=200):
    """Genera un incontro con CPA, severita' e geometria controllati."""
    rng = rng or random
    spec = spec or {
        "type": rng.choice(("Collision", "LoS", "Safe")),
        "angle": rng.randrange(len(ANGLE_BINS)),
        "speed": rng.randrange(len(SPEED_RATIO_BINS)),
        "severity": rng.randrange(3),
        "time": rng.randrange(len(TIME_FRACTION_BINS)),
        "zone": rng.randrange(9),
        "heading": rng.randrange(4),
    }
    for _ in range(max_attempts):
        tipo_obiettivo = spec["type"]
        row, column = divmod(spec["zone"], 3)
        x_inc = rng.uniform((0.20 + 0.20 * column) * AREA_SIZE, (0.40 + 0.20 * column) * AREA_SIZE)
        y_inc = rng.uniform((0.20 + 0.20 * row) * AREA_SIZE, (0.40 + 0.20 * row) * AREA_SIZE)
        z_base = FLIGHT_LEVELS[exp_id % len(FLIGHT_LEVELS)]
        prua0 = rng.uniform(spec["heading"] * 90.0, (spec["heading"] + 1) * 90.0) % 360.0
        relative_angle = rng.uniform(*ANGLE_BINS[spec["angle"]])
        prua1 = (prua0 + relative_angle * rng.choice((-1, 1))) % 360.0
        v0, v1 = _sample_speeds(spec["speed"], rng)
        v0_nm_s, v1_nm_s = v0 * KMH_TO_NM_SEC, v1 * KMH_TO_NM_SEC
        vx0, vy0 = v0_nm_s * math.cos(math.radians(prua0)), v0_nm_s * math.sin(math.radians(prua0))
        vx1, vy1 = v1_nm_s * math.cos(math.radians(prua1)), v1_nm_s * math.sin(math.radians(prua1))
        dvx, dvy = vx0 - vx1, vy0 - vy1
        relative_speed = math.hypot(dvx, dvy)
        if relative_speed < 1e-9:
            continue
        distance = rng.uniform(*SEVERITY_BINS[tipo_obiettivo][spec["severity"]])
        side = rng.choice((-1.0, 1.0))
        nx, ny = side * -dvy / relative_speed, side * dvx / relative_speed
        x0_cpa, y0_cpa = x_inc + nx * distance / 2.0, y_inc + ny * distance / 2.0
        x1_cpa, y1_cpa = x_inc - nx * distance / 2.0, y_inc - ny * distance / 2.0
        if not all(0.0 < value < AREA_SIZE for value in (x0_cpa, y0_cpa, x1_cpa, y1_cpa)):
            continue
        t0_back = calcola_uscita_bordo(x0_cpa, y0_cpa, -vx0, -vy0, AREA_SIZE)[2]
        t1_back = calcola_uscita_bordo(x1_cpa, y1_cpa, -vx1, -vy1, AREA_SIZE)[2]
        time_limit = min(t0_back, t1_back, T_MAX_SIM) * 0.90
        low_fraction, high_fraction = TIME_FRACTION_BINS[spec["time"]]
        low_time, high_time = max(60.0, low_fraction * time_limit), high_fraction * time_limit
        if high_time <= low_time:
            continue
        desired_t_cpa = rng.uniform(low_time, high_time)
        x0_ing, y0_ing = x0_cpa - vx0 * desired_t_cpa, y0_cpa - vy0 * desired_t_cpa
        x1_ing, y1_ing = x1_cpa - vx1 * desired_t_cpa, y1_cpa - vy1 * desired_t_cpa
        t_cpa, dist_cpa_h = calcola_cpa(
            x0_ing, y0_ing, v0_nm_s, prua0, x1_ing, y1_ing, v1_nm_s, prua1
        )
        scenario_effettivo = (
            "Collision" if dist_cpa_h < 0.1
            else "LoS" if dist_cpa_h < SEPARAZIONE_ORIZZONTALE_MIN
            else "Safe"
        )
        if scenario_effettivo != tipo_obiettivo or abs(t_cpa - desired_t_cpa) > 1e-5:
            continue
        x0_usc, y0_usc, t0_usc = calcola_uscita_bordo(x0_ing, y0_ing, vx0, vy0, AREA_SIZE)
        x1_usc, y1_usc, t1_usc = calcola_uscita_bordo(x1_ing, y1_ing, vx1, vy1, AREA_SIZE)
        coverage_stratum = f"{tipo_obiettivo}|a{spec['angle']}|v{spec['speed']}|s{spec['severity']}"
        coverage_cell = (
            f"{coverage_stratum}|t{spec['time']}|z{spec['zone']}|h{spec['heading']}"
        )
        common = [x_inc, y_inc, z_base, t_cpa, scenario_effettivo, t_cpa, dist_cpa_h, 0.0,
                  coverage_stratum, coverage_cell]
        ac0 = [exp_id, 0, x0_ing, y0_ing, z_base, prua0, v0, x0_usc, y0_usc, z_base,
               prua0, v0_nm_s / ROT_MAX_RAD, *common[:3], common[3], t0_usc, *common[4:]]
        ac1 = [exp_id, 1, x1_ing, y1_ing, z_base, prua1, v1, x1_usc, y1_usc, z_base,
               prua1, v1_nm_s / ROT_MAX_RAD, *common[:3], common[3], t1_usc, *common[4:]]
        return ac0, ac1
    raise RuntimeError(f"Impossibile generare lo scenario {exp_id} nello strato {spec}")


def genera_scenari(n_esperimenti, output_file="data/traiettorie_aerei.csv",
                   seed=42, scenario_types=("Collision", "LoS")):
    if n_esperimenti <= 0:
        raise ValueError("n_esperimenti deve essere positivo")
    if not scenario_types or any(value not in SEVERITY_BINS for value in scenario_types):
        raise ValueError("scenario_types deve contenere Collision, LoS e/o Safe")
    rng = random.Random(seed)
    priority = [(index, index % 3, (index // 2) % 3) for index in range(len(ANGLE_BINS))]
    combinations = priority + [
        item for item in product(range(len(ANGLE_BINS)), range(len(SPEED_RATIO_BINS)), range(3))
        if item not in priority
    ]
    strata = [
        (scenario_type, angle, speed, severity)
        for angle, speed, severity in combinations
        for scenario_type in scenario_types
    ]
    coarse_specs = [strata[index % len(strata)] for index in range(n_esperimenti)]
    rng.shuffle(coarse_specs)
    dimensions = {
        "type": [item[0] for item in coarse_specs],
        "angle": [item[1] for item in coarse_specs],
        "speed": [item[2] for item in coarse_specs],
        "severity": [item[3] for item in coarse_specs],
        "time": _balanced_bins(n_esperimenti, len(TIME_FRACTION_BINS), rng),
        "zone": _balanced_bins(n_esperimenti, 9, rng),
        "heading": _balanced_bins(n_esperimenti, 4, rng),
    }
    scenari_data = []
    for exp_id in range(n_esperimenti):
        spec = {name: values[exp_id] for name, values in dimensions.items()}
        ac0, ac1 = calcola_scenari_coerenti(exp_id, spec, rng)
        scenari_data.append(ac0)
        scenari_data.append(ac1)
        
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "ID_Esperimento", "ID_Aereo", 
            "Ingresso_x", "Ingresso_y", "Ingresso_z", 
            "Prua_Iniziale_deg", "Velocita_kmh", 
            "Uscita_Nominale_x", "Uscita_Nominale_y", "Uscita_Nominale_z",
            "Prua_Finale_deg", "RaggioVirataMin_nm",
            "PuntoRiferimento_x", "PuntoRiferimento_y", "PuntoRiferimento_z", "TempoRiferimento_sec", "TempoUscita_sec",
            "Tipo_Scenario", "CPA_Time_sec", "Min_Dist_Horiz_nm", "Min_Dist_Vert_ft",
            "Coverage_Stratum", "Coverage_Cell"
        ])
        writer.writerows(scenari_data)
    manifest = {
        "seed": seed,
        "scenario_count": n_esperimenti,
        "scenario_types": list(scenario_types),
        "counts": {
            name: dict(Counter(values))
            for name, values in dimensions.items()
        },
        "scenarios": [
            {"experiment_id": index, **{name: values[index] for name, values in dimensions.items()}}
            for index in range(n_esperimenti)
        ],
    }
    manifest_path = f"{os.path.splitext(output_file)[0]}_coverage.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Generati {n_esperimenti} scenari stratificati in: {output_file}")
    print(f"Manifest copertura: {manifest_path}")

def leggi_scenari_csv(input_file):
    scenari = {}
    with open(input_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            exp_id = int(row["ID_Esperimento"])
            if exp_id not in scenari:
                scenari[exp_id] = []
            scenari[exp_id].append({
                "id_aereo": int(row["ID_Aereo"]),
                "p0": (float(row["Ingresso_x"]), float(row["Ingresso_y"]), float(row["Ingresso_z"])),
                "prua0": float(row["Prua_Iniziale_deg"]),
                "v0": float(row["Velocita_kmh"]),
                "p_target": (float(row["Uscita_Nominale_x"]), float(row["Uscita_Nominale_y"]), float(row["Uscita_Nominale_z"])),
                "prua_f": float(row["Prua_Finale_deg"]),
                "r_min": float(row["RaggioVirataMin_nm"]),
                "t_reference": float(row["TempoRiferimento_sec"]),
                "exit_time_sec": float(row["TempoUscita_sec"]),
                "tipo": row["Tipo_Scenario"],
                "t_cpa": float(row["CPA_Time_sec"]),
                "dist_h": float(row["Min_Dist_Horiz_nm"]),
                "dist_v": float(row["Min_Dist_Vert_ft"]),
                "coverage_stratum": row.get("Coverage_Stratum", "legacy"),
                "coverage_cell": row.get("Coverage_Cell", "legacy"),
            })
    return scenari

def visualizza_scenari(input_file):
    import matplotlib.pyplot as plt

    scenari = leggi_scenari_csv(input_file)
    colori = ['blue', 'orange']
    
    for exp_id, aerei in scenari.items():
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        tipo_scen = aerei[0]["tipo"]
        t_cpa = aerei[0]["t_cpa"]
        
        for aereo in aerei:
            path = genera_traiettoria_nominale_4d(
                aereo["p0"][0], aereo["p0"][1], aereo["p0"][2],
                aereo["prua0"], aereo["v0"], t_max=T_MAX_SIM
            )
            x_vals = [p[0] for p in path]
            y_vals = [p[1] for p in path]
            z_vals = [p[2] for p in path]
            
            ax.plot(x_vals, y_vals, z_vals, label=f"AC{aereo['id_aereo']} ({int(aereo['v0'])} km/h)", color=colori[aereo['id_aereo']], linewidth=2)
            ax.scatter(x_vals[0], y_vals[0], z_vals[0], color=colori[aereo['id_aereo']], s=60)
            
            # Evidenzia la posizione dell'aereo all'istante di massima vicinanza (CPA)
            idx_cpa = int(t_cpa // DT)
            if idx_cpa < len(x_vals):
                ax.scatter(x_vals[idx_cpa], y_vals[idx_cpa], z_vals[idx_cpa], marker='o', color='red', s=40)
        
        # Colore del titolo dinamico basato sul livello di rischio
        color_title = 'red' if tipo_scen == 'Collision' else 'darkorange' if tipo_scen == 'LoS' else 'green'
        
        ax.set_xlim(0, AREA_SIZE)
        ax.set_ylim(0, AREA_SIZE)
        ax.set_zlim(min(FLIGHT_LEVELS) - 2000, max(FLIGHT_LEVELS) + 2000)
        ax.set_xlabel('Spazio X (NM)')
        ax.set_ylabel('Spazio Y (NM)')
        ax.set_zlabel('Altitudine Z (Piedi)')
        
        titolo = (f"Scenario {exp_id} - [{tipo_scen.upper()}]\n"
                  f"CPA al sec {int(t_cpa)} | Distanza Orizzontale: {aerei[0]['dist_h']:.2f} NM | Separazione Verticale: {aerei[0]['dist_v']:.0f} ft")
        ax.set_title(titolo, color=color_title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True)
        
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generatore Scenari ATM Dinamici e Generalizzati")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--num", type=int, default=scn.get("n_experiments", 1200))
    gen_parser.add_argument("--output", default=scn.get("input_scenarios_file", "data/traiettorie_aerei.csv"))
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--types", nargs="+", choices=tuple(SEVERITY_BINS), default=["Collision", "LoS"])
    
    vis_parser = subparsers.add_parser("visualize")
    vis_parser.add_argument("--input", type=str, default="data/traiettorie_aerei.csv")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        os.makedirs("data", exist_ok=True)
        genera_scenari(args.num, args.output, args.seed, tuple(args.types))
    elif args.command == "visualize":
        try:
            visualizza_scenari(args.input)
        except FileNotFoundError:
            print(f"Errore: Il file {args.input} non esiste. Esegui prima il comando 'generate'.")

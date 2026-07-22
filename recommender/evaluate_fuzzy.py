import json
import numpy as np
import glob
import os
from stable_baselines3 import PPO
from typing import List, Dict
from rlfuzzy import FuzzyDecisionMaker
from rlfuzzy import FuzzyDecisionMaker, GenotypeEncoder


class ModelEvaluator:
    """Valuta un modello addestrato su nuovi dati di Pareto front"""
    
    def __init__(self, model_path: str, fuzzy_decider: FuzzyDecisionMaker):
        """
        Args:
            model_path: percorso del modello salvato (.zip)
            fuzzy_decider: istanza del sistema fuzzy
        """
        self.model = PPO.load(model_path)
        self.fuzzy_decider = fuzzy_decider
        self.genotype_encoder = GenotypeEncoder(max_waypoints=10)
        print(f"Modello caricato da: {model_path}")
    
    def load_test_scenarios(self, test_folder: str) -> List[Dict]:
        """
        Carica gli scenari di test da una nuova directory
        
        Args:
            test_folder: directory contenente i file *_pareto_front.json
        
        Returns:
            lista di scenari di test
        """
        scenarios = []
        file_list = glob.glob(os.path.join(test_folder, "*_pareto_front_*.json"))
        
        if not file_list:
            raise ValueError(f"Nessun file trovato in {test_folder}")
        
        print(f"Trovati {len(file_list)} file di test")
        
        for idx, filename in enumerate(file_list):
            try:
                with open(filename) as f:
                    d = json.load(f)
                    scenario = d.get("scenario", {})
                    population = d.get("population", [])
                    
                    if not population or not scenario:
                        continue
                    
                    pareto_front = []
                    genotypes = []
                    
                    for p in population:
                        fitness = p.get("fitness", [])
                        genome = p.get("genome", [])
                        
                        if len(fitness) < 3 or not genome:
                            continue
                        
                        # Mantieni i valori reali (senza normalizzazione)
                        pareto_front.append({
                            "path_length": fitness[0],
                            "exit_distance": fitness[1],
                            "exit_direction": fitness[2]
                        })
                        genotypes.append(genome)
                    
                    if not pareto_front:
                        continue
                    
                    scenario_struct = {
                        "scenario_id": idx,
                        "filename": os.path.basename(filename),
                        "collision_time": scenario.get("collision_time", 0),
                        "collision_distance": scenario.get("area_size", 0),
                        "relative_velocity": scenario.get("speed_late", 0),
                        "pareto_front": pareto_front,
                        "genotypes": genotypes
                    }
                    scenarios.append(scenario_struct)
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Errore nel caricamento di {filename}: {e}")
                continue
        
        print(f"Caricati {len(scenarios)} scenari validi")
        return scenarios
    
    def _create_observation(self, scenario: Dict, selected_idx: int = None) -> np.ndarray:
        """
        Crea un'osservazione dall'scenario (deve corrispondere all'environment di training)
        
        Args:
            scenario: dizionario dello scenario
            selected_idx: indice della soluzione selezionata (opzionale)
        
        Returns:
            osservazione come numpy array
        """
        pareto = scenario['pareto_front']
        
        avg_path_length = np.mean([s['path_length'] for s in pareto])
        avg_exit_distance = np.mean([s['exit_distance'] for s in pareto])
        avg_exit_direction = np.mean([s['exit_direction'] for s in pareto])
        
        base_obs = np.array([
            scenario['collision_time'],
            scenario['collision_distance'],
            scenario['relative_velocity'],
            avg_path_length,
            avg_exit_distance,
            avg_exit_direction,
            len(pareto)
        ], dtype=np.float32)
        
        # Aggiungi encoding del genotipo (se presente)
        if 'genotypes' in scenario and len(scenario['genotypes']) > 0:
            all_stats = [self.genotype_encoder.encode_genotype_stats(g) 
                        for g in scenario['genotypes']]
            genotype_features = np.mean(all_stats, axis=0)
        else:
            genotype_features = np.zeros(8, dtype=np.float32)
        
        # Allineamento con l'ambiente: aggiungi 3 feature per la soluzione (qui messe a zero)
        sol_features = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([base_obs, genotype_features, sol_features])
        return obs
    
    def evaluate_scenario(self, scenario: Dict, epsilon: float = 0.0, 
                         deterministic: bool = True) -> Dict:
        """
        Valuta un singolo scenario
        
        Args:
            scenario: dizionario dello scenario
            epsilon: epsilon per il fuzzy decider (0 = sempre greedy)
            deterministic: se True, usa azioni deterministiche dal modello
        
        Returns:
            dizionario con i risultati della valutazione
        """
        pareto = scenario['pareto_front']
        
        # 1. Predizione del modello RL
        obs = self._create_observation(scenario)
        action, _states = self.model.predict(obs, deterministic=deterministic)
        action = int(np.clip(action, 0, len(pareto) - 1))
        
        # 2. Scelta del sistema fuzzy (ground truth)
        fuzzy_idx, all_preferences = self.fuzzy_decider.select_solution(
            pareto, epsilon=epsilon
        )
        
        # 3. Calcola metriche
        match = (action == fuzzy_idx)
        
        suggested_sol = pareto[action]
        fuzzy_sol = pareto[fuzzy_idx]
        
        # Distanza tra la soluzione suggerita e quella fuzzy
        distance = np.sqrt(
            ((suggested_sol['path_length'] - fuzzy_sol['path_length']) / 113.0) ** 2 +
            ((suggested_sol['exit_distance'] - fuzzy_sol['exit_distance']) / 160.0) ** 2 +
            ((suggested_sol['exit_direction'] - fuzzy_sol['exit_direction']) / 360.0) ** 2
        )
        
        # Differenza di preferenza
        pref_diff = abs(all_preferences[action] - all_preferences[fuzzy_idx])
        
        return {
            "scenario_id": scenario['scenario_id'],
            "filename": scenario['filename'],
            "pareto_size": len(pareto),
            "model_action": action,
            "fuzzy_action": fuzzy_idx,
            "match": match,
            "distance": distance,
            "preference_diff": pref_diff,
            "model_preference": all_preferences[action],
            "fuzzy_preference": all_preferences[fuzzy_idx],
            "all_preferences": all_preferences,
            "suggested_solution": suggested_sol,
            "fuzzy_solution": fuzzy_sol
        }
    
    def evaluate_all(self, test_folder: str, epsilon: float = 0.0,
                    deterministic: bool = True) -> Dict:
        """
        Valuta il modello su tutti gli scenari di test
        
        Args:
            test_folder: directory con i file di test
            epsilon: epsilon per il fuzzy decider
            deterministic: se True, usa azioni deterministiche
        
        Returns:
            dizionario con le metriche aggregate
        """
        scenarios = self.load_test_scenarios(test_folder)
        
        if not scenarios:
            raise ValueError("Nessuno scenario di test caricato")
        
        results = []
        matches = []
        distances = []
        pref_diffs = []
        
        print(f"\nValutazione su {len(scenarios)} scenari...")
        
        for i, scenario in enumerate(scenarios):
            result = self.evaluate_scenario(scenario, epsilon, deterministic)
            results.append(result)
            matches.append(result['match'])
            distances.append(result['distance'])
            pref_diffs.append(result['preference_diff'])
            
            if (i + 1) % 10 == 0:
                print(f"  Valutati {i+1}/{len(scenarios)} scenari...")
        
        # Calcola metriche aggregate
        accuracy = np.mean(matches) * 100
        avg_distance = np.mean(distances)
        avg_pref_diff = np.mean(pref_diffs)
        
        summary = {
            "num_scenarios": len(scenarios),
            "accuracy": accuracy,
            "num_matches": sum(matches),
            "avg_distance": avg_distance,
            "std_distance": np.std(distances),
            "avg_preference_diff": avg_pref_diff,
            "std_preference_diff": np.std(pref_diffs),
            "detailed_results": results
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Stampa un riepilogo dei risultati"""
        print("\n" + "="*70)
        print("RISULTATI DELLA VALUTAZIONE")
        print("="*70)
        print(f"Numero di scenari testati: {summary['num_scenarios']}")
        print(f"Accuracy (match con fuzzy): {summary['accuracy']:.2f}%")
        print(f"Numero di match: {summary['num_matches']}/{summary['num_scenarios']}")
        print(f"\nDistanza media dalle soluzioni fuzzy: {summary['avg_distance']:.4f} "
              f"(± {summary['std_distance']:.4f})")
        print(f"Differenza di preferenza media: {summary['avg_preference_diff']:.2f} "
              f"(± {summary['std_preference_diff']:.2f})")
        print("="*70)
    
    def save_results(self, summary: Dict, output_file: str):
        """Salva i risultati in un file JSON evitando riferimenti circolari"""

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj


        # Copia profonda e conversione
        clean_summary = convert(summary)

        with open(output_file, 'w') as f:
            json.dump(clean_summary, f, indent=2)

        print(f"\nRisultati salvati in: {output_file}")


# Script principale di valutazione
def evaluate_model(model_path: str, test_folder: str, output_file: str = None):
    """
    Funzione principale per valutare il modello
    
    Args:
        model_path: percorso del modello salvato
        test_folder: directory con i nuovi dati di Pareto
        output_file: file dove salvare i risultati (opzionale)
    """
    # Crea il fuzzy decider (deve essere lo stesso del training)
    fuzzy_decider = FuzzyDecisionMaker()
    
    # Crea l'evaluator
    evaluator = ModelEvaluator(model_path, fuzzy_decider)
    
    # Valuta su tutti gli scenari
    summary = evaluator.evaluate_all(
        test_folder=test_folder,
        epsilon=0.0,  # Usa sempre la scelta greedy del fuzzy
        deterministic=True  # Usa azioni deterministiche dal modello
    )
    
    # Stampa i risultati
    evaluator.print_summary(summary)
    
    # Salva i risultati se specificato
    if output_file:
        evaluator.save_results(summary, output_file)
    
    return summary


# Esempio di utilizzo
if __name__ == "__main__":
    # Valuta il modello su una nuova directory
    summary = evaluate_model(
        model_path="pareto_recommender_multi_scenario_model.zip",
        test_folder="../data",  # Directory con i nuovi Pareto front
        output_file="evaluation_results.json"
    )
    
    # Analisi dettagliata (opzionale)
    print("\n--- Top 5 scenari con maggiore differenza di preferenza ---")
    detailed = summary['detailed_results']
    sorted_by_pref = sorted(detailed, key=lambda x: x['preference_diff'], reverse=True)
    
    for i, result in enumerate(sorted_by_pref[:5]):
        print(f"\n{i+1}. Scenario: {result['filename']}")
        print(f"   Match: {result['match']}")
        print(f"   Differenza preferenza: {result['preference_diff']:.2f}")
        print(f"   Modello: {result['suggested_solution']}")
        print(f"   Fuzzy: {result['fuzzy_solution']}")

import numpy as np

from pymoo.core.result import Result
from model_ga.population import PopulationExtended as Population
from model_ga.individual import IndividualSimulated as Individual
from utils.sorting import *
import dill
import os
from pathlib import Path
from visualization import output

RESULTS_FOLDER = os.sep + "results" + os.sep
WRITE_ALL_INDIVIDUALS = True

class SimulationResult(Result):

    def __init__(self) -> None:
        super().__init__()

    def obtain_history(self):
        hist = self.history
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation
            for algo in hist:
                n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
                opt = algo.opt  # retrieve the optimum from the algorithm
                feas = np.where(opt.get("feasible"))[
                    0]  # filter out only the feasible and append and objective space values
                hist_F.append(opt.get("F")[feas])
        else:
            n_evals = None
            hist_F = None
        return n_evals, hist_F

    def obtain_all_population(self):
        all_population = Population()
        hist = self.history
        for generation in hist:
            all_population = Population.merge(all_population, generation.pop)
        return all_population

    def obtain_history_hitherto(self):
        hist = self.history
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation

        opt_all = Population()
        for algo in hist:
            n_evals.append(algo.evaluator.n_eval)
            opt_all = Population.merge(opt_all, algo.pop)
            opt_all_nds = get_nondominated_population(opt_all)
            feas = np.where(opt_all_nds.get("feasible"))[
                0]
            hist_F.append(opt_all_nds.get("F")[feas])
        return n_evals, hist_F
    
    def persist(self, save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(save_folder + os.sep + "result", "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(save_folder, name="result"):
        with open(save_folder + os.sep + name, "rb") as f:
            return dill.load(f)
            
    def write_results(self, results_folder = RESULTS_FOLDER):
        algorithm = self.algorithm
        algorithm_name = algorithm.__class__.__name__        

        # if self is None:
        #     print("Result object is None. Execute algorithm first, before writing results.")
        #     return
        print(f"=====[{algorithm_name}] Writing results...")

        # config = self.config
        # algorithm_parameters = {
        #     "Population size" : str(config.population_size),
        #     "Number of generations" : str(config.n_generations),
        #     "Number of offsprings": str(config.num_offsprings),
        #     "Crossover probability" : str(config.prob_crossover),
        #     "Crossover eta" : str(config.eta_crossover),
        #     "Mutation probability" : str(config.prob_mutation),
        #     "Mutation eta" : str(config.eta_mutation)
        # }

        save_folder = output.create_save_folder(self.problem, results_folder, algorithm_name)

        output.convergence_analysis(self, save_folder)
        output.hypervolume_analysis(self, save_folder)
        output.spread_analysis(self, save_folder)
        
        output.write_calculation_properties(self,save_folder,algorithm_name)
        output.design_space(self, save_folder)
        output.objective_space(self, save_folder)
        output.optimal_individuals(self, save_folder)
        output.write_summary_results(self, save_folder)
        output.write_simulation_output(self,save_folder)
        output.simulations(self, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(self, save_folder)

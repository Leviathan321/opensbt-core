from algorithm.SimAlgo import SimAlgo
from model_ga.problem import *
from model_ga.result import ResultExtended
from simulation.simulator import *
from algorithm.algorithm import *
from experiment.search_configuration import *
from visualization import output

RESULTS_FOLDER = os.sep + "results" + os.sep
WRITE_ALL_INDIVIDUALS = True

@dataclass
class Experiment(object):
    # TODO refactor nsag2/nsga2-DT into a class to use algorithm instance in algorithm parameter
        
    def __init__(self, problem: Problem, algorithm: SimAlgo, search_configuration: SearchConfiguration):
        self.problem = problem
        self.algorithm = algorithm
        self.search_configuration = search_configuration

    def run(self) -> ResultExtended:
        if self.algorithm is not None:
            return self.algorithm.run()
        return None

    def write_results(self, results_folder = RESULTS_FOLDER):
        algorithm = self.algorithm
        algorithm_name = algorithm.algorithm_name        
        res = self.res

        if self.res is None:
            print("Result object is None. Execute algorithm first, before writing results.")
            return
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

        save_folder = output.create_save_folder(res.problem, results_folder, algorithm_name)

        output.convergence_analysis(res, save_folder)
        output.hypervolume_analysis(res, save_folder)
        output.spread_analysis(res, save_folder)
        output.write_calculation_properties(res,save_folder,algorithm_name,algorithm.config.__dict__)
        output.design_space(res, save_folder)
        output.objective_space(res, save_folder)
        output.optimal_individuals(res, save_folder)
        output.write_summary_results(res, save_folder)
        output.write_simulation_output(res,save_folder)
        output.simulations(res, save_folder)

        if WRITE_ALL_INDIVIDUALS:
            output.all_individuals(res, save_folder)

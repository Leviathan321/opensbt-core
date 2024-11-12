import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm import nsga2_optimizer
from opensbt.algorithm import nsga2dt_optimizer

from default_experiments import *
from opensbt.experiment.search_configuration import *
from default_experiments import *
import logging as log
from opensbt.analysis.Analysis import Analysis
from opensbt.utils.log_utils import *
from opensbt.config import *
from opensbt.experiment.experiment_store import experiments_store
from opensbt.config import METRIC_CONFIG
from opensbt.simulation.dummy_simulation import DummySimulator
from tests import test_base

class TestAnalysisMulit():
    def test_analysis_objective(self):
        test_base.configure_logging_and_env()

        class CriticalAdasDistanceVelocityTest(Critical):
            def eval(self, vector_fitness, simout: SimulationOutput = None):
                if simout is not None:
                    isCollision = simout.otherParams['isCollision']
                else:
                    isCollision = None

                if(isCollision == True) or (vector_fitness[0] < 2) and (vector_fitness[1] < -1):
                    return True
                else:
                    return False
                
        problem = ADASProblem(
                            problem_name="DummySimulatorProblem",
                            scenario_path="./dummy_scenario",
                            xl=[0, 1, 0, 1],
                            xu=[360, 3,360, 3],
                            simulation_variables=[
                                "orientation_ego",
                                "velocity_ego",
                                "orientation_ped",
                                "velocity_ped"],
                            fitness_function=FitnessMinDistanceVelocity(),
                            critical_function=CriticalAdasDistanceVelocityTest(),
                            simulate_function=DummySimulator.simulate,
                            simulation_time=10,
                            sampling_time=0.25
                            )
        
        config = DefaultSearchConfiguration()
        config.population_size = 50
        config.n_generations = 20
        class_algos = [  
            nsga2_optimizer.NsgaIIOptimizer,
            nsga2dt_optimizer.NsgaIIDTOptimizer
        ]
        algo_names = [
            "NSGA-II",
            "NSGA-II-DT"
        ]
        n_runs = 5
        n_func_evals_lim = 200 
        distance_tick = 0.1*n_func_evals_lim

        ###################
        problems = []
        configs = []

        problems = [problem, problem]

        config_1 = DefaultSearchConfiguration()
        config_1.population_size = 10
        config_1.n_generations = 20

        ####################

        config_2 = DefaultSearchConfiguration()
        config_2.population_size = 10
        #config_2.ref_points = np.asarray([[0,-4]])
        config_2.inner_num_gen = 5
        config_2.n_func_evals_lim = n_func_evals_lim
        
        configs = [config_1,config_2]

        ideal = METRIC_CONFIG["DUMMY"]["ideal"]
        ref_point_hv =  METRIC_CONFIG["DUMMY"]["ref_point_hv"]
        nadir =  ref_point_hv

        path_metrics = None
        analysis_name = None
        output_folder = os.getcwd() + os.sep + "tests" + os.sep + "output" + os.sep + "analysis"
        folder_runs = None
        crit_function = None

        Analysis.run(
                    analysis_name = analysis_name,
                    algo_names = algo_names,
                    class_algos = class_algos,
                    configs = configs,
                    n_runs = n_runs,
                    problems = problems,
                    n_func_evals_lim = n_func_evals_lim, 
                    folder_runs = folder_runs,
                    path_metrics = path_metrics,
                    ref_point_hv = ref_point_hv,
                    ideal = ideal,
                    nadir = nadir,
                    output_folder = output_folder,
                    do_coverage_analysis = False,
                    do_ds_analysis = True,
                    path_critical_set = None,
                    debug = DEBUG,
                    distance_tick = distance_tick,
                    do_evaluation = True,
                    crit_function = crit_function,
                    color_map=None
        )
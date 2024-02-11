import pymoo

from opensbt.model_ga.individual import IndividualSimulated
from opensbt.analysis.quality_indicators.quality import Quality

pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2dt_optimizer import NsgaIIDTOptimizer
from opensbt.evaluation.critical import CriticalBnhDivided
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.problem.pymoo_test_problem import *
import matplotlib.pyplot as plt
import os

from opensbt.visualization import output_metric, visualizer

WAIT_RESULTS_TIME = 10

OUTPUT_FOLDER  = os.getcwd() + os.sep + "tests" + os.sep + "output" + os.sep


class TestAnalysis():

 
    def test_coverage_analysis(self):
        pass
    def test_convergence_analysis(self):
        pass

    def test_analysis_math(self):

        problem = PymooTestProblem(
            'BNH', critical_function= CriticalBnhDivided())

        config = DefaultSearchConfiguration()

        config.population_size = 50
        config.inner_num_gen = 2
        config.prob_mutation = 0.5
        config.n_func_evals_lim = 1000

        config.ideal = np.asarray([0,0])
        config.ref_point_hv = np.asarray([150,50])
        config.nadir = config.ref_point_hv

        optimizer = NsgaIIDTOptimizer(problem,config)

        res = optimizer.run()

        #########################        
        save_folder = visualizer.create_save_folder(problem, 
                                                    results_folder=OUTPUT_FOLDER, 
                                                    algorithm_name = optimizer.algorithm_name, 
                                                    is_experimental=False)

        res.write_results(results_folder=save_folder)

        ######### Evaluate

        bound_min = config.ideal
        bound_max = config.nadir
        n_cells = 10

        output_metric.hypervolume_analysis(res, 
                save_folder, 
                nadir = config.nadir,
                ideal = config.ideal
                )
        
        output_metric.hypervolume_analysis_local(res, 
                save_folder
                )
        
        output_metric.calculate_n_crit_distinct(res,
                                                save_folder=save_folder,
                                                bound_min = bound_min,
                                                bound_max=bound_max,
                                                n_cells=n_cells,
                                                var="F")
                
        output_metric.calculate_n_crit_distinct(res,
                                                save_folder=save_folder,
                                                bound_min = bound_min,
                                                bound_max=bound_max,
                                                n_cells=n_cells,
                                                var="X")
        output_metric.igd_analysis(res, 
                save_folder, 
                input_pf= None
                )
        
        output_metric.igd_analysis_hitherto(res, 
            save_folder, 
            input_pf= None
        )
        
        
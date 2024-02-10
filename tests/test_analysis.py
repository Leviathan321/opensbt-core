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

from opensbt.visualization import output_metric

class TestAnalysis():

    WAIT_RESULTS_TIME = 10
    OUTPUT_FOLDER  = os.getcwd() + "tests" + os.sep + "output" + os.sep
    
    def test_coverage_analysis(self):
        pass
    def test_convergence_analysis(self):
        pass

    def test_hv_analysis_math(self):
        problem = PymooTestProblem(
            'BNH', critical_function= CriticalBnhDivided())

        config = DefaultSearchConfiguration()

        config.population_size = 50
        config.inner_num_gen = 2
        config.prob_mutation = 0.5
        config.n_func_evals_lim = 1000

        config.ideal = np.asarray([0,0])
        config.ref_point_hv = np.asarray([70,20])
        config.nadir = config.ref_point_hv

        optimizer = NsgaIIDTOptimizer(problem,config)

        res = optimizer.run()
        res.write_results()

        ######### Evaluate

        all_pop = res.obtain_all_population()
        crit_pop = all_pop.divide_critical_non_critical()[0]

        eval_result = Quality.calculate_hv_hitherto(res, crit_pop.get("X"))
        n_evals, digd = eval_result.steps, eval_result.values


        plt.figure(figsize=(7, 5))
        plt.plot(n_evals, digd, color='black', lw=0.7)
        plt.scatter(n_evals, digd, facecolor="none", edgecolor='black', marker='o')
        plt.title("Design Space Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("HV")
        plt.savefig(os.getcwd() + os.sep + "quality_indicators" + os.sep + "test.png")
        plt.close()

        output_metric.hypervolume_analysis(res, 
                save_folder=TestAnalysis.OUTPUT_FOLDER + os.sep + "quality_indicators" + os.sep, 
                nadir =    config.nadir,
                ideal = config.ideal
                )
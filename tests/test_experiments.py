import os
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.evaluation.critical import CriticalAdasDistanceVelocity
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.evaluation.fitness import *
from opensbt.problem.adas_problem import ADASProblem


class TestExperiments():

    def test_dummy_experiment(self):
        from opensbt.simulation.dummy_simulation import DummySimulator
        
        results_folder = '/tests/output/'

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
                            fitness_function=FitnessMinDistanceVelocityFrontOnly(),
                            critical_function=CriticalAdasDistanceVelocity(),
                            simulate_function=DummySimulator.simulate,
                            simulation_time=10,
                            sampling_time=0.25
                          )
        config = DefaultSearchConfiguration()
        config.population_size = 20
        config.n_generations = 20

        print("pymoo NSGA-II algorithm is used.")
        
        algo = NsgaIIOptimizer(
                              problem=problem,
                              config=config)
        res = algo.run()        
        res.write_results(results_folder=results_folder,
                        params = algo.parameters)

from typing import List
from scenario import Scenario
from models.scenario import ScenarioInstance
from optimization.optimizer import Optimizer

class NSGAII_Optimizer(Optimizer):
    
    @staticmethod
    def findOptimalTestCases(scenario: Scenario) -> List[ScenarioInstance]:
        pass
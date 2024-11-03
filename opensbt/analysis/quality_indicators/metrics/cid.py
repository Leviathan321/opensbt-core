from opensbt.analysis.quality_indicators. \
     utils.distance_indicator_sets import DistanceIndicator, euclidean_distance

class CID(DistanceIndicator):
    """ This class implements the coverage inverted distance (CID) metric [1] to evaluate how well enough search algorithm can cover failures. 
    
       [1] Sorokin, Lev et al. “Can Search-Based Testing with Pareto Optimization Effectively Cover Failure-Revealing Test Inputs?” (EMSE, 2024).
    """
    def __init__(self, Z, **kwargs):
        super().__init__(Z, euclidean_distance, 1, **kwargs)

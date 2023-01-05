class SearchConfiguration(object):
    n_generations = None 
    population_size = None
    maximal_execution_time = None
    max_tree_iterations = None
    num_offsprings = None
    prob_crossover = None
    eta_crossover = None
    prob_mutation = None
    eta_mutation = None
    # NSGAII-DT specific
    inner_num_gen = None

#TODO create a search configuration file specific for each algorithm
class DefaultSearchConfiguration(SearchConfiguration):
    n_generations = 5 
    population_size = 20
    maximal_execution_time = None
    max_tree_iterations = 4
    num_offsprings = None
    prob_crossover = 0.7
    eta_crossover = 20
    prob_mutation = None
    eta_mutation = 15
    # NSGAII-DT specific
    inner_num_gen = 4
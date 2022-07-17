from datetime import datetime
from distutils.log import error
from matplotlib.pyplot import ylim
from sklearn import tree
import numpy as np
import copy
import pydotplus  
import os
import csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG = False

MIN_SAMPLES_SPLIT = 200
CRITICALITY_THRESHOLD = 0.9
DELTA = 0.0

def getCriticalRegions(all_solutions, all_critical_dict, var_min, var_max,  name, feature_names, outputPath=None, criticality_probability = CRITICALITY_THRESHOLD, min_samples_split=MIN_SAMPLES_SPLIT, plot_results=True):

        X = all_solutions
        y = list(all_critical_dict.values())        
        solutions_in_region = []


        assert np.array([str(all_solutions[i]) == list(all_critical_dict.keys())[i]  for i in range(len(all_solutions))]).all()
        assert( len(all_solutions) == len(y))

        if all(elem == 1 for elem in y):
            print("all candidates are critical")
            bounds = [(var_min,var_max)]
            solutions_in_region = [all_solutions]  
            return solutions_in_region, bounds
        elif (all(elem == 0 for elem in y)):
            print("all candidates are non critical")
            bounds = []
            solutions_in_region = []           
            return solutions_in_region, bounds

        CP = criticality_probability
        clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, criterion="entropy", max_depth=10)
        clf = clf.fit(X, y)

        tree.plot_tree(clf)

        r = tree.export_text(clf)

        if DEBUG:
            print(r)

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        n_node_samples= clf.tree_.n_node_samples
        values= clf.tree_.value

        if DEBUG:
            print("n_nodes: "+ str(n_nodes))
            print("child left : " + str(children_left))
            print("child right : " +str(children_right))
            print("feature: "+ str(feature))
            print("threshold: "+ str(threshold))
            print("n_node_samples: "+ str(n_node_samples))
            print("values: " + str(values))

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        leafsLib = {}
        criticalReg = []

        indCrit = 1
        indNotCri = 0

        for i in range(0,n_nodes):
            if is_leaves[i]:
                values_node_i =  values[i][0]
                if values_node_i[indCrit] / ( values_node_i[indNotCri] +  values_node_i[indCrit]) > CP:
                    criticalReg.append(i)

        if DEBUG:
            print("Critical regsions: " + str(criticalReg))

        classResults = clf.apply(X)

        def ind_in_node(node_id):
                inds = []
                for i in range(0,len(classResults)):
                    if classResults[i] == node_id:
                        inds.append(i)
                return inds

        # get features from critical regions to explore regions
        bounds= []
        delta = DELTA
        pop_regions = []

        node_indicator = clf.decision_path(X)
        leave_id = classResults

        for i in criticalReg:
            individuals_in_region = ind_in_node(i)
            # print(len(individuals_in_region))
            pop_regions.append(individuals_in_region)

            #sample in class i 
            upperReg = copy.deepcopy(var_max)
            lowerReg = copy.deepcopy(var_min)
            
            sample_id = individuals_in_region[0]

            node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

            print("node_index: " + str(node_index))
            print("ind: " + str(X[sample_id]))

            for node_id in node_index:
                if leave_id[sample_id] == node_id:  # <-- changed != to ==
                    #continue # <-- comment out
                    print("leaf node {} reached, no decision here".format(leave_id[sample_id])) # <--
                else: # < -- added else to iterate through decision nodes
                    if (X[sample_id][feature[node_id]] <= threshold[node_id]):
                        threshold_sign = "<=" 
                        j = node_id
                        # set upper bound
                        # in a DT nodes might have same threshold features
                        if threshold[j] < upperReg[feature[j]]:  
                            upperReg[feature[j]] = threshold[j]
                        if DEBUG:
                            print("at left: "+ str(j))
                            print("feature {} <= {}".format(feature[j],threshold[j]))
                    else:
                        threshold_sign = ">"
                        j = node_id
                        # set lower bound
                        if threshold[j] > lowerReg[feature[j]]:  
                            lowerReg[feature[j]] = threshold[j] + DELTA
                        if DEBUG:
                            if node_id == 0:
                                print("at root node")
                            print("at right: "+ str(j))
                            print("feature {} > {}".format(feature[j],threshold[j]))
                        
                    print("decision id node %s : (X[%s, %s] (= %s) %s %s)"  % (node_id, 
                            sample_id,
                            feature[node_id],
                            X[sample_id][feature[node_id]], 
                            threshold_sign,
                            threshold[node_id]))

                                      # validate
            assert np.less(np.array(lowerReg) , np.array(upperReg)).all
           
            bounds.append((lowerReg,upperReg))

        for i in range(0,len(bounds)):
            print("bound for region {} is {}".format(i,bounds[i]))
            #print("individuals of the region are : \n{}".format(pop_regions))

        # get solutions instead only indices as output
        for pop_region_ind in pop_regions:
            temp = [all_solutions[ind] for ind in pop_region_ind]
            solutions_in_region.append(temp)

        assert len(bounds) == len(solutions_in_region)

        if plot_results:
            print("++ Plot decision tree ++")
            dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True,  # leaves_parallel=True, 
                                special_characters=True,feature_names=feature_names)  
            graph = pydotplus.graph_from_dot_data(dot_data)  
            #date = int(round(datetime.now().timestamp()))
            date =  datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            if outputPath == None:
                dir = CURRENT_DIR  + os.path.sep +  "results" + os.path.sep + date  
            else:
                dir = outputPath
            if not os.path.isdir(dir):
                os.makedirs(dir)
            graph.write_pdf(dir +os.sep + name + "_regions.pdf" )

            print("++ Write critical regions ++")

                    # write report of execution

            filename_bounds = dir +os.sep  + "bounds_regions.csv"
            if not os.path.isfile(filename_bounds):
                header = ['region', 'bounds'] 

                with open(filename_bounds, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header) 
                    for i in range(len(bounds)):
                        writer.writerow([f"region {str(i)}" , bounds[i]])
                    writer.writerow(["",""])
            else:
                with open(filename_bounds, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    for i in range(len(bounds)):
                        writer.writerow([f"region {str(i)}" , bounds[i]])
                    writer.writerow(["",""])

            # TODO

        ''' Debugging '''
        for i in range(len(solutions_in_region)):
            for sol in solutions_in_region[i]:
                try:
                    assert np.less_equal(np.array(sol) ,np.array(bounds[i][1])).all()
                    assert np.less_equal(np.array(bounds[i][0]),np.array(sol)).all()
                except AssertionError as e:
                    print(sol)
                    print(bounds[i])
                    ind = list(all_critical_dict.keys()).index(str(sol))
                    print("index of solution: " + str(ind))
                    raise e

        return solutions_in_region, bounds

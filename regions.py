from datetime import datetime
from distutils.log import error
from tkinter import Y
from matplotlib.pyplot import ylim
from sklearn import tree
import numpy as np
import copy
import pydotplus  
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG = True

def getCriticalRegions(X,y, var_min, var_max, criticality_probability = 0.6):
        if all(elem == 1 for elem in y):
            print("all candidates are critical")
            bounds = [(var_min,var_max)]
            pop_regions = [range(0,len(X))]  
            return pop_regions, bounds
        elif (all(elem == 0 for elem in y)):
            print("all candidates are non critical")
            bounds = []
            pop_regions = []           
            return pop_regions, bounds
            
        feature_names= ["x_ego", "y_ego", "angle_ego","v_ego","x_ped", "y_ped", "angle_ped","v_ped"]

        CP = criticality_probability
        pop_regions = 0

        clf = tree.DecisionTreeClassifier(min_samples_split=20)
        clf = clf.fit(X, y)

        tree.plot_tree(clf)

        r = tree.export_text(clf)

        if DEBUG:
            dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True,  # leaves_parallel=True, 
                                special_characters=True,feature_names=feature_names)  
            graph = pydotplus.graph_from_dot_data(dot_data)  
            #date = int(round(datetime.now().timestamp()))
            date =  datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            graph.write_pdf(CURRENT_DIR  + os.path.sep +  "results" + os.path.sep + "crit_regions_{}.pdf".format(date))

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
            # print(node.get_label())
            # print(node.get_name())
            if  is_leaves[i]:
                valuesNodeI = values[i][0]
                leafsLib[i] = {"critical" : 0, "notCritical" : 0}
                
                # issue: if all are critical/non critical only one numbre is stored
                if (len(valuesNodeI) > 1):
                    leafsLib[i]["critical"]= valuesNodeI[indCrit]
                    leafsLib[i]["notCritical"] = valuesNodeI[indNotCri]
                else:
                    leafsLib[i]["critical"]= valuesNodeI[0]
                leafsLib[i]["notCritical"] = valuesNodeI[indNotCri]
                if(leafsLib[i]["critical"]/(leafsLib[i]["notCritical"] +leafsLib[i]["critical"]) > CP):
                    criticalReg.append(i)

        def getParentsTree(ch_left, ch_right):
            result = {}
            for i in range(0,len(ch_left)):
                if(ch_left[i]!=-1):
                    result[ch_left[i]] = i

            for i in range(0,len(ch_right)):
                if(ch_right[i]!=-1):
                    result[ch_right[i]] = i
            return result

        parents = getParentsTree(children_left,children_right)

        def getPredecessors(node_id):
            result = []
            node = node_id
            while node != 0:
                result.append(parents[node])
                node = parents[node]
            return result

        def ind_in_node(node_id, pop):
                inds = []
                classResults = clf.apply(pop)
                for i in range(0,len(classResults)):
                    if classResults[i] == node_id:
                        inds.append(i)
                return inds

        # get features from critical regions to explore regions
        bounds= []
        delta = 0.001
        pop_regions = []

        for i in criticalReg:
            individuals_in_region = ind_in_node(i,pop=X)
            pop_regions.append(individuals_in_region)

            #sample in class i 
            upperReg = copy.deepcopy(var_max)
            lowerReg = copy.deepcopy(var_min)

            for j in getPredecessors(i):
                if(j in children_left and j == 0 ):
                    upperReg[feature[j]] = threshold[j]
                    #print("at: "+ str(j))

                    #print("feature {} <= {}".format(feature[j],threshold[j]))
                else:
                    lowerReg[feature[j]] = threshold[j] + delta
                    #print("at: "+ str(j))
                    #print("feature {} > {}".format(feature[j],threshold[j]))

            # validate
            assert np.less_equal(np.array(lowerReg)[0] , np.array(upperReg)[0])

            bounds.append((lowerReg,upperReg))

  

        print("n cr regions:" + str(len(criticalReg)))

        for i in range(0,len(bounds)):
            print("bound for region {} is {}".format(i,bounds[i]))
            #print("individuals of the region are : \n{}".format(pop_regions))
            
        return pop_regions, bounds

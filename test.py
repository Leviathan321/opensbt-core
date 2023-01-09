import importlib
#module_name= "experiment.experiment"

#modules= ["pymoo.algorithms.moo.nsga2"]
module_name = "pymoo.algorithms.moo.nsga2"

class_name = "Individual"
MyClass = getattr(importlib.import_module(module_name), class_name)
instance = MyClass()
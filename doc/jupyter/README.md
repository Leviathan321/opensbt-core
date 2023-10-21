# Tutorials

In this folder several tutorials are provided of how to use OpenSBT and apply if for a specific testing problem. The tutorials can be directly executed using the Jupyter Notebook environment. 

Following commands are required to install the Jupyter Notebook environment (On Linux, where Python3.8 is installed):

```python -m pip install virtualenv```
```python -m virtualenv venv```

Activate virtual environment:

```source venv/bin/activate```
```python -m pip install ipykernel```

Register virtual environment at kernel:

```python3 -m ipykernel install --user --name=venv```

Start jupyter notebook:

```jupyter notebook```

Select *Kernel > Change Kernel > venv*.

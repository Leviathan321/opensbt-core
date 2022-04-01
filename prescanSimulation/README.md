# Running PRESCAN experiments from python

## Prerequisites

1. Start MATLAB via PRESCAN Process Manager
2. Expose MATLAB Engine by running in MATLAB console:

```
matlab.engine.shareEngine
```

3. Run experiment script based on prescanWorker.py with a python 3.7 interpreter (newer versions are not supported by the matlab engine)
# DNN Lane Keeping Testing with Donkey

This example integrates the case study into OpenSBT presented here: https://github.com/testingautomated-usi/maxitwo.

To run the case study you have to Download the Donkey simulator from this [link](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Fsimulators), and a dnn model of your choice from [here](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Flogs%2Fmodels).

You have to update then the DNN_MODEL_PATH and DONKEY_EXE_PATH variable in the config file of this folder.

Install then the requirements by:

```bash
pip install -r examples/lanekeeping/requirements.txt
```

You can execute the test case generation with:

```bash
python -m examples.lanekeeping.donkey.run_donkey
```
# DNN Lane Keeping Testing with Donkey

This example integrates the case study into OpenSBT presented here: https://github.com/testingautomated-usi/maxitwo.

To run the case study you have to Download the Donkey simulator from this [link](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Fsimulators), and a dnn model of your choice from [here](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ?path=%2Flogs%2Fmodels).

Install then the requirements by:

```bash
pip install -r exmaples/lanekeeping/requirements.txt
```

You can execute the test case generation with:

```bash
python -m examples.lanekeeping.donkey.run_donkey
```
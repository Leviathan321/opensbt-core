python -m pip install virtualenv
python -m virtualenv venv
source venv/bin/activate
python -m pip install ipykernel
python3 -m ipykernel install --user --name=venv

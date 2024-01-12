from setuptools import setup, find_packages

setup(
    name='opensbt',
    version='0.1.3',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'deap==1.4.1',
        'numpy==1.24.4',
        'scipy==1.10.1',
        'matplotlib==3.7.4',
        'docker==6.1.3',
        'pandas==2.0.3',
        'setuptools',
        'scikit-learn==1.3.2',
        'pymoo==0.6.0.1'
    ],
    python_requires=">=3.8"
)

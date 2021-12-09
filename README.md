# Machine Learning for Argument-Based Computational Persuasion
This repo contains the source code and a benchmark for predicting user's utilities with Machine Learning techniques for Computational Persuasion



CODE WILL BE SOON AVAILABLE




## Requirements

The following packages are required:

-   [numpy](http://www.numpy.org/), tested with version 1.19.5
-   [matplotlib](http://matplotlib.org/), tested with version 3.3.4
-   [sklearn](https://scikit-learn.org/stable/), tested with version 0.24.1
-   [pandas](https://pandas.pydata.org/), tested with version 1.1.5

## Usage

Type:
```
 $ python3 experiments_util_prediction_parallel.py -p ${parallelism_flag}
```
where `${parallelism_flag}` can be `True` or `False` whether you want to run the experiments using all the available CPUs in your machine.

To run the simulations. Type:
```
 $ python3 meat_example_experiments.py
```
to run the experiment with the red meat case study.

## Files

- `data/DT` contains the decision trees of the simulations;
- `data/datasets` contains the datasets of the simulations;
- `results` contains the results of the simulations;
- `results/tree_samples` contains the results for each tree of the simulations;
- `meat_data` contains the input data for the red meat case study;
- `meat_results` contains the results for the red meat case study;

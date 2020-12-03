# Light Random Forest (lightrf)
 Experimental Random Forest for Handling Extremely Large Dataset based on Scikit-Learn.

## Background
Random Forest (RF) in Scikit-Learn is arguably the most efficient implementation we have now. However, its training costs are still prohibitively large on extremely large datasets. For instance, more than 50GB memory is required in order to fit a `sklearn.ensemble.RandomForestClassifier` with only 100 decision trees on the HIGGS dataset (~10 million samples).

This repository contains my personal attempt on implementing a reduced version of random forest/decision tree in Scikit-Learn, which has much smaller training and evaluating costs than the original version in Scikit-Learn, especially in terms of the usage on running memory.

To achieve this, some basic ideas are:
* Implement classic teachniques for accelerating decision tree, e.g., data binning;
* Remove properties irrelevant to the model inference on decision tree, e.g., feature importance;
* Simplify the underlying data structure of decision tree.

## Installation

```
$ git clone https://github.com/AaronX121/Light-Random-Forest.git
$ cd Light-Random-Forest
$ pip install -r requirements.txt
$ python setup.py install
```

* Please see the script in `./examples` for details on how to use.
* As a kind reminder, **an additional stage on data binning is required by lightrf**, while the remaining workflow is exactly the same as using the Random Forest in Scikit-Learn.

## Experiment Results
* **Covtype Dataset**
    * 581,012 samples | 54 features
    * n_estimator=500 | n_jobs=-1 | Remaining hyper-parameters were set to their default values
* **HIGGS Dataset**
    * 11,000,000 samples | 28 features
    * n_estimator=100 | n_jobs=-1 | Remaining hyper-parameters were set to their default values
* **More Results**
    * Each numerical cell in the table below denoted the results of LightRF / Scikit-Learn RF, respectively.
    * Curves in the figure below were reported by Memory Profiler (https://github.com/pythonprofilers/memory_profiler)
    * Similar results could be reproduced by running `./examples/classification_comparison.py`

|      Metric Name    |   COVTYPE   |     HIGGS     |
|:-------------------:|:-----------:|:-------------:|
|   Testing Acc (%)   | 73.74/73.74 |  75.86/75.86  |
|   Model Size (GB)   |  1.45/4.02  |   5.31/18.67  |
|  Training Time (s)  | 19.87/25.46 | 653.47/749.10 |
| Evaluating Time (s) |  1.56/1.90  |   3.01/4.77   |

![Experiment Results](./experiment.png)

## Package dependencies
* joblib>=0.11
* Cython>=0.28.5
* scikit-learn>=0.22

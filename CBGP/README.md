# Plexicase Selection in Code Building Genetic Programming

This implementation is built on the top of [Code Building Genetic Programming](https://github.com/erp12/CodeBuildingGeneticProgramming-ProtoType) by Eddie Pantridge.


## Experiments

The experiments are tested with Python3.8.

To run experiments with lexicase selection:
```
cd CBGP
python3 run_lexicase.py exp_id downsample_rate
```

`exp_id` will be splitted into two parts, where
```
problem_id = exp_id // 100
    exp_id = exp_id % 100
```

So a full run of experiments will use `exp_id` from 0 to 699, which consists of 7 PSB problems with 100 trials for each problem. 

`downsample_rate` is a float number from 0-1, which uses random downsampling on the training set during each generation. In the paper, we use `downsample_rate` of 1 and 0.25.

Similarly, to run experiments with plexicase selection:
```
python3 run_lexiprob.py exp_id downsample_rate alpha
```

`alpha` is a float number from 0-inf, which is a hyperparameter we introduced to tune the shape of probability distribution of selection. Lower `alpha` gives more uniform distributions, and higher values gives more weights on elite solutions. 

## Results
`vis_results.ipynb` offers some functions to view and visualize the results produced by the experiments, including solution rates and overlaps. 

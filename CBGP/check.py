import json
import numpy as np
import glob

exps = glob.glob('results/*')

results = {}
for exp in exps:
    files = glob.glob(exp+'/*')
    exp_name = exp.split('/')[-1]
    results[exp_name] = []
    for file in files:
        with open(file, 'r') as f:
            results[exp_name].append(json.load(f))

for exp in sorted(results.keys()):
    success = sum([i['test_err']==0 for i in results[exp]])
    print(exp, success)
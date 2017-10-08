import numpy as np
import matplotlib.pyplot as plt
from util import generate_data, generate_data2, random_walk, label_propagation, gaussian_kernel, hyperparam_opt_random_walk, hyperparam_opt_label_prop

#Now plot the effect of more data

nums_unlabeled = [10, 30, 50, 100, 500, 1000]
verbose = False

for i,n_unlabeled in enumerate(nums_unlabeled):
    print('Plotting iter %i of %i'%(i,len(nums_unlabeled)))
    with open('log/%s.txt'%n_unlabeled, 'a') as f:
        # Evaluate random walk
        trials = 10
        runs_per_trial = 10
        best_MAE = 1000
        for trial in range(trials):
            # Generate data
            n_labeled = 10
            n_train = 100
            # n_test = 200
            X_labeled, y_labeled = generate_data2(n_labeled)
            X_unlabeled, _ = generate_data2(n_unlabeled)
            X_train, y_train = generate_data2(n_train)
            # X_test, y_test = generate_data2(n_test)
            for run in range(runs_per_trial):
                std = 0.01+np.random.rand()/4

                try:
                    y_pred = random_walk(y_labeled, X_labeled, X_unlabeled, X_train, gaussian_kernel(std))
                except np.linalg.linalg.LinAlgError:
                    y_pred = np.zeros_like(y_train)
                MAE = np.mean(np.abs(y_pred-y_train))
                accuracy = np.mean(np.equal(y_pred > 0.0, y_train > 0.0))
                if MAE > 1.0:
                    continue
                f.write('%5.3f,%5.3f\n'%(std, MAE))

fig,axarr = plt.subplots(len(nums_unlabeled),1)
for ax in axarr: ax.set_ylim(0,1)
for i,n_unlabeled in enumerate(nums_unlabeled):
    with open('log/%s.txt'%n_unlabeled, 'r') as f:
        for line in f:
            data = list(map(float, line.split(',')))
            axarr[i].scatter(data[0],data[1])

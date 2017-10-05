import numpy as np
import matplotlib.pyplot as plt
from util import generate_data, generate_data2, random_walk, label_propagation, gaussian_kernel, hyperparam_opt_random_walk, hyperparam_opt_label_prop, make_subplots

#Make a dataset
n_unlabeled, n_labeled, n_train = 500, 100, 100

X_labeled, y_labeled = generate_data2(n_labeled)
X_unlabeled, y_unlabeled = generate_data2(n_unlabeled)
X_train, y_train = generate_data2(n_train)

#Plot some data
total_data = np.concatenate((X_labeled, X_unlabeled, X_train), 0)
total_targets = np.concatenate((y_labeled, y_unlabeled, y_train), 0)
plt.scatter(total_data[:,0], total_data[:,1], c=total_targets)

"""Run the algorithms"""

#Evaluate Random Walk
y_pred = random_walk(y_labeled, X_labeled, X_unlabeled, X_train, gaussian_kernel(0.1))
MAE = np.mean(np.abs(y_pred-y_train))
accuracy = np.mean(np.equal(y_pred > 0.0, y_train > 0.0))
print('We have MAE %5.3f and accuracy %5.3f'%(MAE, accuracy))

#Evaluate Label Prop
y_pred = label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, gaussian_kernel(0.1), mu = 0.4)
MAE = np.mean(np.abs(y_pred-y_train))
accuracy = np.mean(np.equal(y_pred > 0.0, y_train > 0.0))
print('We have MAE %5.3f and accuracy %5.3f'%(MAE, accuracy))

"""Experiment with different sizes of unlabeled data"""

nums_unlabeled = [10, 30, 50, 100, 500, 1000]

fig, axarr = make_subplots(2,2)

for i,n_unlabeled in enumerate(nums_unlabeled):
    print('Plotting iter %i of %i'%(i,len(nums_unlabeled)))
    for trial in range(5):
        # Generate data
        n_labeled = 10
        n_train = 100
        n_test = 200
        X_labeled, y_labeled = generate_data2(n_labeled)
        X_unlabeled, _ = generate_data2(n_unlabeled)
        X_train, y_train = generate_data2(n_train)
        X_test, y_test = generate_data2(n_test)

        # Evaluate label prop
        y_pred = label_propagation(y_labeled, X_labeled, X_unlabeled, X_test, gaussian_kernel(0.15), mu = 0.5)
        MAE = np.mean(np.abs(y_pred-y_test))
        accuracy = np.mean(np.equal(y_pred > 0.0, y_test > 0.0))
        # n_unlabeled  = float(n_unlabeled)
        axarr[0,0].scatter(n_unlabeled, MAE,c='r')
        axarr[1,0].scatter(n_unlabeled, accuracy,c='r')
        print('For labelprop, at num_unlabeled %6i we have MAE %5.3f Acc %5.3f'%(n_unlabeled, MAE, accuracy))

        # Evaluate random walk
        y_pred = random_walk(y_labeled, X_labeled, X_unlabeled, X_test, gaussian_kernel(0.1))
        MAE = np.mean(np.abs(y_pred-y_test))
        accuracy = np.mean(np.equal(y_pred > 0.0, y_test > 0.0))
        axarr[0,1].scatter(n_unlabeled,MAE,c='b')
        axarr[1,1].scatter(n_unlabeled,accuracy,c='b')
        print('For randomwalk, at num_unlabeled %6i we have MAE %5.3f Acc %5.3f'%(n_unlabeled, MAE, accuracy))




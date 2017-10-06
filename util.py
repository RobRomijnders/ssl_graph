import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from copy import copy

def make_subplots(rows, cols):
    """
    Just a utility function to manage labeling all the axis
    :param rows: number of rows in subplots
    :param cols: number of columks in subplots
    :return:
    """
    #Prepare the subplots
    f, axarr = plt.subplots(rows, cols)
    for axrow in axarr:
        for ax in axrow:
            ax.set_xscale('log')
            ax.set_xlabel('num unlabeled samples')
            ax.set_ylim(0,1)

    axarr[0,0].set_ylabel('Mean absolute erros')
    axarr[0,0].set_title('Labelprop')
    axarr[1,0].set_ylabel('Accuracy')

    axarr[0,1].set_ylabel('Mean absolute erros')
    axarr[0,1].set_title('Randomwalk')
    axarr[1,1].set_ylabel('Accuracy')
    return f, axarr


def gaussian_kernel(std=1.3):
    """
    Makes a kernel function with fixed standard deviation
    :param std: length scale of the  Gaussian kernel
    :return:
    """
    def kernel(x, y=None):
        # Discriminate two cases
        # - when only x is given, the Gram matrix will be symmetric. We can use this fact to save computation
        # - when both x and y are given, then do normal calculation
        if y is None:
            d = squareform(pdist(x))
        else:
            d = cdist(x, y)
        K = np.exp(-1/2*np.square(d/std))
        # Possibly, you can zero-out the contributions from non-neighbors
        # D = K.shape[1]
        # neighbors = 5
        # idx = np.argpartition(K,D-neighbors, axis=1)[:,:D-neighbors]
        # for i,id in enumerate(idx):
        #     K[i,id] = 0
        return K
    return kernel

def generate_data(N):
    """
    data generating function for the two half circles
    :param N:
    :return:
    """
    #Sample random angle
    angle = np.random.rand(N)*5-1

    #Sample random radius
    radius_scale = 0.3
    radius = np.random.rand(N)*2*radius_scale+1.0-radius_scale

    #Convert angle and radius to Carthesian coordinates
    x0 = radius*np.cos(angle)+0.5
    x1 = radius*np.sin(-1*angle)

    lbl = 2*np.random.randint(0,2,size=N)-1

    data = np.vstack((x0,x1)).T

    # Multiply with the labels to get the mirror version of the original half circle
    data = data*np.expand_dims(lbl,1)
    return data,lbl

def generate_data2(N):
    """
    data generating function for the spiral with decreasingly growing radius
    :param N:
    :return:
    """
    alpha_list = [0, float(np.pi)]
    t_max = 10 # max time for which to run the radius

    data = []
    targets = []
    for i, alpha in enumerate(alpha_list):
        N_half = int(N/2)
        t = t_max*np.random.rand(N_half,) #add some noise to the radius :)
        radius = np.sqrt(t)+0.05*np.random.randn(N_half,)
        data.append(np.vstack((radius*np.cos(t+alpha), radius*np.sin(t+alpha))).T)
        targets.append(i*np.ones((N_half,)))

    #Permute the data
    perm = np.random.permutation(N)
    data = np.concatenate(data,0)[perm]
    targets = np.concatenate(targets,0)[perm]
    return data, 2*targets-1


def random_walk(y_labeled, X_labeled, X_unlabeled, X_train, kernel):
    """
    Runs a random walk by calculating the P_infinity matrix. it assumes a model
    where all the labeled points will
    be absorbing states, so P(i->j)=1. The unlabeled points will not have self connections,
    so P(i->j)=0.
    This results in a block structure like
    [[P_ll, P_lu],[P_ul,P_uu]] = [[identity_matrix, zero_matrix],[P_ul,P_uu]]
    :param y_labeled: targets for labeled points
    :param X_labeled:  data for labeled points
    :param X_unlabeled: data for unlabeled points
    :param X_train: data for where to evaluate the label
    :param kernel: kernel function
    :return:
    """
    n_unlabeled = X_unlabeled.shape[0]
    n_labeled = X_labeled.shape[0]
    n_train = X_train.shape[0]

    # from here on, we define X_eval as the data points where we want to EVALuate the labels
    X_eval = np.concatenate((X_unlabeled, X_train), 0)

    # Subfixes refer to the data. _e refers to eval data, _l refers to the labeled data
    # So p_el refers to transition probability from eval data to labeled data
    P_el = kernel(X_eval, X_labeled)
    P_ee = kernel(X_eval, None)

    # Set diagonal of P_ee to zero
    # P_ee -= np.diagonal(P_ee)

    # Normalize the rows to ensure each row represents a transition distribution
    row_sum = np.expand_dims(np.sum(P_el,1)+np.sum(P_ee,1),1)
    P_el /= row_sum
    P_ee /= row_sum

    # Calculate the lower left block of the infinity matrix
    # P_inf = np.linalg.inv(np.eye(n_unlabeled+n_train) - P_ee).dot(P_el)
    P_inf = np.linalg.solve(np.eye(n_unlabeled+n_train) - P_ee, P_el)

    assert P_inf.shape == (n_unlabeled+n_train, n_labeled) #Just to debug the code

    #Calculate the predictions on the final n_train samples
    y_pred = np.dot(P_inf[-n_train:], y_labeled)
    return y_pred

def label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, kernel, mu=1., verbose=False):
    """
    Label propagation algorithm on the data
    :param y_labeled: targets for labeled points
    :param X_labeled:  data for labeled points
    :param X_unlabeled: data for unlabeled points
    :param X_train: data for where to evaluate the label
    :param kernel: kernel function
    :param mu: hyperparameter for the label prop algo
    :param verbose: do you want to print stuff?
    :return:
    """
    n_unlabeled = X_unlabeled.shape[0]
    n_labeled = X_labeled.shape[0]
    n_train = X_train.shape[0]

    # from here on, we define X_eval as the data points where we want to EVALuate the labels
    X_eval = np.concatenate((X_labeled, X_unlabeled, X_train), 0)

    # Subfixes refer to the data. _e refers to eval data, _l refers to the labeled data
    # So p_el refers to transition probability from eval data to labeled data
    W = kernel(X_eval, None)
    D = np.sum(W,0)

    eps = 1E-9 #arbitrary small number

    A = np.diag(np.concatenate((np.ones((n_labeled)), np.zeros((n_unlabeled+n_train)))) + mu*D + mu*eps)

    y_hat_0 = np.concatenate((y_labeled, np.zeros((n_unlabeled+n_train))))
    y_hat = copy(y_hat_0)

    for iter in range(100):
        y_hat_old = y_hat
        y_hat = np.linalg.solve(A, mu*np.dot(W, y_hat)+y_hat_0)

        if np.linalg.norm(y_hat - y_hat_old) < 0.01:
            if verbose:
                print('Converged after %i steps'%iter)
            break
    else:
        if verbose:
            print('Not converged??')

    return y_hat[-n_train:]


def hyperparam_opt_label_prop(y_labeled, X_labeled, X_unlabeled, X_train, y_train, verbose = False):
    ## OLD CODE
    trials = 10
    best_MAE = 1000
    for trial in range(trials):
        mu = np.random.rand()
        std = 0.01+np.random.rand()/2

        y_pred = label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, gaussian_kernel(std), mu = mu, verbose=verbose)
        MAE = np.mean(np.abs(y_pred-y_train))
        accuracy = np.mean(np.equal(y_pred > 0.0, y_train > 0.0))

        if MAE < best_MAE:
            if verbose:
                print('At mu %5.3f and std %5.3f we have MAE %5.3f (and acc %5.3f)'%(mu,std, MAE, accuracy))
            best_MAE = MAE
            best_hyper_param = (std, mu)

    return best_hyper_param

def hyperparam_opt_random_walk(y_labeled, X_labeled, X_unlabeled, X_train, y_train, verbose = False):
    ## OLD CODE
    trials = 10
    best_MAE = 1000
    for trial in range(trials):
        std = 0.01+np.random.rand()/2

        try:
            y_pred = random_walk(y_labeled, X_labeled, X_unlabeled, X_train, gaussian_kernel(std))
        except np.linalg.linalg.LinAlgError:
            y_pred = np.zeros_like(y_train)
        MAE = np.mean(np.abs(y_pred-y_train))
        accuracy = np.mean(np.equal(y_pred > 0.0, y_train > 0.0))

        if MAE < best_MAE:
            if verbose:
                print('At std %5.3f we have MAE %5.3f (and acc %5.3f)'%(std, MAE, accuracy))
            best_MAE = MAE
            best_hyper_param = std
    return best_hyper_param
from cgi import test
import numpy as np
import os

from image import export_image

def load_data(train = True):
    """
    Load the data from disk

    Parameters
    ----------
    train : bool
        Load training data if true, else load test data
    Returns
    -------
        Tuple:
            Images
            Labels
    """
    directory = 'train' if train else 'test'
    patterns = np.load(os.path.join('./data/', directory, 'images.npz'))['arr_0']
    labels = np.load(os.path.join('./data/', directory, 'labels.npz'))['arr_0']
    return patterns.reshape(len(patterns), -1), labels

def z_score_normalize(X, u = None, xd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    if u == None:
        u = np.mean(X, axis = 0)
    
    if xd == None:
        xd = np.std(X, axis = 0)

    return (X - u) / xd, (u, xd)

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    if _min == None:
        _min = np.min(X, axis=0)

    if _max == None:
        _max = np.max(X, axis=0)

    return (X - _min) / (_max - _min), (_min, _max)

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    k = np.max(y) + 1
    return np.eye(k)[y]

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis = 1)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)rain
            Labels (y)
    """
    X, y = dataset
    order = np.random.permutation(len(X))
    return X[order], y[order]

def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape (N,(d+1))
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))

def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset, k = 10): 
    X, y = dataset

    order = np.random.permutation(len(X))
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width

def get_binary_subset(trainset, testset, class_0_and_6 = True):
    X_train, y_train = trainset
    X_test, y_test = testset

    if class_0_and_6 == True:
    # Get the indices of class 0 (o) and class 2 (ma)
        trainset_mask = np.where((y_train == 0) | (y_train == 2))[0]
        testset_mask = np.where((y_test == 0) | (y_test == 2))[0]
        X_train_masked = X_train[trainset_mask]
        y_train_masked = (y_train[trainset_mask] == 2).astype(int)
        X_test_masked = X_test[testset_mask]
        y_test_masked = (y_test[testset_mask] == 2).astype(int)
    else:
    # Get the indices of class 2 (ma) and class 6 (su)
        trainset_mask = np.where((y_train == 2) | (y_train == 6))[0]
        testset_mask = np.where((y_test == 2) | (y_test == 6))[0]
        X_train_masked = X_train[trainset_mask]
        y_train_masked = (y_train[trainset_mask] == 6).astype(int)
        X_test_masked = X_test[testset_mask]
        y_test_masked = (y_test[testset_mask] == 6).astype(int)

    return (X_train_masked, y_train_masked), (X_test_masked, y_test_masked)

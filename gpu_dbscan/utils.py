# Print iterations progress
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    #print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_test_blobs(n_samples=1000, d=2):
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.1, random_state=0)
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)
    return X

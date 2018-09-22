import numpy as np


class MNIST(object):
    def __init__(self, label):
        dataset = {}
        data = np.load('data/mnist.npz')

        full_x = np.concatenate([data['x_train'], data['x_test'], data['x_valid']], axis=0)
        full_y = np.concatenate([data['y_train'], data['y_test'], data['y_valid']], axis=0)

        inlier_x = full_x[full_y == label]
        inlier_y = full_y[full_y == label]

        n_inlier = inlier_x.shape[0]
        shuf = np.random.permutation(n_inlier)
        inlier_x = inlier_x[shuf]
        inlier_y = inlier_y[shuf]

        train_x = inlier_x[:-(n_inlier // 3)]
        test_inlier_x = inlier_x[-(n_inlier // 3):]
        test_inlier_y = inlier_y[-(n_inlier // 3):]

        outlier_x = full_x[full_y != label]
        outlier_y = full_y[full_y != label]

        n_outlier = outlier_x.shape[0]
        shuf = np.random.permutation(n_outlier)
        outlier_x = outlier_x[shuf]
        outlier_y = outlier_y[shuf]

        n_test = test_inlier_x.shape[0] * 2
        test_outlier_x = outlier_x[:n_test]
        test_outlier_y = outlier_y[:n_test]

        dataset['train'] = train_x
        dataset['test'] = (
            np.concatenate([test_inlier_x, test_outlier_x], 0),
            np.concatenate([test_inlier_y, test_outlier_y], 0),
        )
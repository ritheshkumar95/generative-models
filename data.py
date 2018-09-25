import numpy as np
import torch
from torchvision.utils import save_image


class MNIST(object):
    def __init__(self, label):
        self.dataset = {}
        data = np.load('anomaly_data/mnist.npz')

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
        test_inlier_y = np.zeros_like(inlier_y[-(n_inlier // 3):])

        outlier_x = full_x[full_y != label]
        outlier_y = full_y[full_y != label]

        n_outlier = outlier_x.shape[0]
        shuf = np.random.permutation(n_outlier)
        outlier_x = outlier_x[shuf]
        outlier_y = outlier_y[shuf]

        n_test = test_inlier_x.shape[0] * 2
        test_outlier_x = outlier_x[:n_test]
        test_outlier_y = np.ones_like(outlier_y[:n_test])

        self.dataset['train'] = train_x
        self.dataset['test'] = (
            np.concatenate([test_inlier_x, test_outlier_x], 0),
            np.concatenate([test_inlier_y, test_outlier_y], 0),
        )

    def inf_train_gen(self, batch_size):
        while True:
            idxs = np.random.randint(
                0, self.dataset['train'].shape[0], batch_size
            )
            x = self.dataset['train'][idxs].reshape((-1, 1, 28, 28))
            yield torch.from_numpy(2 * x - 1).cuda()

    def test_gen(self, batch_size):
        data = self.dataset['test']
        for i in range(0, data[0].shape[0], batch_size):
            x = data[0][i:i + batch_size].reshape((-1, 1, 28, 28))
            y = data[1][i:i + batch_size]
            x = torch.from_numpy(2 * x - 1).cuda()
            y = torch.from_numpy(y).cuda()
            yield x, y.long()


if __name__ == '__main__':
    dataset = MNIST(1)
    itr = dataset.inf_train_gen(64)
    x = itr.__next__()
    save_image(x, 'train_mnist.png', normalize=True)

    test_itr = dataset.test_gen(64)
    for i in range(100):
        x, y = test_itr.__next__()
    save_image(x, 'test_mnist.png', normalize=True)
    print(y)

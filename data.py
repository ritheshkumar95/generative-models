from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import torch


class SwissRoll(object):
    def __init__(self, n_data, std):
        self.n_data = n_data
        self.std = std
        self.data = make_swiss_roll(n_data, std)[0].astype('float32')
        self.data /= self.data.max(0, keepdims=True)

        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 2])
        plt.savefig('orig_samples.png')

    def create_iterator(self, batch_size):
        data = torch.from_numpy(self.data[:, [0, 2]]).cuda()
        for i in range(self.n_data // batch_size - 1):
            yield data[i * batch_size: (i + 1) * batch_size]

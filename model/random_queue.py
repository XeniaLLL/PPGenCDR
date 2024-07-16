import numpy as np


class Random_queue(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity # batch_size * i_factor_size
        self.batch_size = batch_size
        self.length = 0
        self.data = []
        self.label = []

    def set_data(self, user_indices, user_ratings_real):

        if len(self.data) == 0:
            shape = user_indices.shape[1:]
            self.data = np.zeros([self.capacity] + list(shape))
            shape = user_ratings_real.shape[1:]
            self.label = np.zeros([self.capacity]+ list(shape))

        if self.length < self.capacity: # note  queue 不合理
            for i in range(user_indices.shape[0]):
                self.data[self.length] = user_indices[i:i + 1] # 抽取batch 中top capacity
                self.label[self.length] = user_ratings_real[i:i + 1]
                self.length += 1
                if self.length >= self.capacity: # careful rethink
                    break
        else:
            permutation = np.random.permutation(self.length) # careful 为什么queue 还要随机排序            for i in range(user_indices.shape[0]):
            for i in range(user_indices.shape[0]):
                self.data[permutation[i]] = user_indices[i]
                self.label[permutation[i]] = user_ratings_real[i]

    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # print(batch_size, self.length)
        if batch_size > self.length:
            return self.data[:self.length], self.label[:self.length]

        results, results_l = [], []
        permutation = np.random.permutation(self.length)
        for i in range(batch_size):
            results.append(self.data[permutation[i]])
            results_l.append(self.label[permutation[i]])
        user_indices = np.stack(results, 0).astype(np.int64)
        user_ratings = np.stack(results_l, 0).astype(np.float32)
        return user_indices, user_ratings


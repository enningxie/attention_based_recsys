import numpy as np
from scipy.sparse import dok_matrix


class DataSet(object):
    def __init__(self, path):
        self.trainMatrix = self.train_rating_to_matrix(path + '.train.rating')
        self.testRatings = self.test_rating_to_list(path + '.test.rating')
        self.testNegatives = self.test_load_negtive_file(path + '.test.negative')
        assert len(self.testRatings) == len(self.testNegatives), 'error with dataSet.'
        self.num_users, self.num_items = self.trainMatrix.shape

    def test_rating_to_list(self, filename):
        rating_list = []
        with open(filename, 'r') as f:
            data_rating = f.readlines()
        for rating in data_rating:
            arr = rating.split('\t')
            user_id, item_id = int(arr[0]), int(arr[1])
            rating_list.append([user_id, item_id])
        return rating_list

    def test_load_negtive_file(self, filename):
        negtive_list = []
        with open(filename, 'r') as f:
            data_neg = f.readlines()
        for negtive in data_neg:
            arr = negtive.split('\t')
            negtives = []
            for x in arr[1:]:
                negtives.append(int(x))
            negtive_list.append(negtives)
        return negtive_list

    def train_rating_to_matrix(self, filename):
        '''
        read .rating file and return dok matrix
        :param filename:
        :return:
        '''
        num_users, num_items = 0, 0
        with open(filename, 'r') as f:
            data_rating = f.readlines()
        for rating in data_rating:
            arr = rating.split('\t')
            uer_id, item_id = int(arr[0]), int(arr[1])
            num_users = max(num_users, uer_id)
            num_items = max(num_items, item_id)

        # construct matrix
        mat = dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for rating in data_rating:
            arr = rating.split('\t')
            user_id, item_id, rating_= int(arr[0]), int(arr[1]), float(arr[2])
            if rating_ > 0:
                mat[user_id, item_id] = 1.0
        return mat
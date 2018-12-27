from src.Dataset import DataSet
import os

TMP_DATA_PATH = '../Data'

if __name__ == '__main__':
    tmp_dict = {2: 0.6, 1: 0.2, 8: 0.9, 6: 0.7}
    print(sorted(tmp_dict, key=tmp_dict.get, reverse=True)[: 2])

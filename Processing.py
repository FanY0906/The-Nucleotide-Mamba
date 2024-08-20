import numpy as np




class Processing():
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y


    def __len__(self):
        return len(self.data_X)


    def convert(self, single_seq):
        datas = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 4, 'N':5, '_':6}
        single_code = []
        for x in single_seq:
            single_code.append(datas[x])
        return np.asarray(single_code)

    def __getitem__(self, idx):
        x = self.data_X[idx]
        x = self.convert(x)
        y = self.data_Y[idx]


        return x, y

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class msrcv1(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MSRCv1.mat')['Y'].astype(np.int32).reshape(210, )
        self.V1 = scipy.io.loadmat(path + 'MSRCv1.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MSRCv1.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MSRCv1.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'MSRCv1.mat')['X4'].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'MSRCv1.mat')['X5'].astype(np.float32)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),
                torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class scene(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'scene.mat')['Y'].astype(np.int32).reshape(2688, )
        self.V1 = scipy.io.loadmat(path + 'scene.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'scene.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'scene.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'scene.mat')['X4'].astype(np.float32)

    def __len__(self):
        return 2688

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class handwritten(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'handwritten.mat')['Y'].astype(np.int32).reshape(2000, )
        self.V1 = scipy.io.loadmat(path + 'handwritten.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'handwritten.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'handwritten.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'handwritten.mat')['X4'].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'handwritten.mat')['X5'].astype(np.float32)
        self.V6 = scipy.io.loadmat(path + 'handwritten.mat')['X6'].astype(np.float32)

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "MNIST-USPS":
        dataset = MNIST_USPS('../DATA/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "msrcv1":
        dataset = msrcv1('../DATA/')
        dims = [24, 576, 512, 256, 254]
        view = 5
        class_num = 7
        data_size = 210
    elif dataset == "handwritten":
        dataset = handwritten('../DATA/')
        dims = [240, 76, 216, 47, 64,6]
        view = 6
        class_num = 10
        data_size = 2000
    elif dataset == "scene":
        dataset = scene('../DATA/')
        dims = [512, 432, 256,48]
        view = 4
        class_num = 8
        data_size = 2688
    elif dataset == "Caltech-2V":
        dataset = Caltech('../DATA/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('../DATA/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('../DATA/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('../DATA/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

def load_test(dataset):
    if dataset == "Caltech-2V":
        params = {'dataname': "Caltech-2V", "feature_dim": 768,
                  "high_feature_dim": 768, "mid_dim": 512,
                  "layers1": 2, "layers2": 5, "chance": 3}
    elif dataset == "Caltech-3V":
        params = {'dataname': "Caltech-3V", "feature_dim": 1024,
                  "high_feature_dim": 1024, "mid_dim": 1024,
                  "layers1": 3, "layers2": 4, "chance": 3}
    elif dataset == "Caltech-4V":
        params = {'dataname': "Caltech-4V", "feature_dim": 512,
                  "high_feature_dim": 2048, "mid_dim": 512,
                  "layers1": 2, "layers2": 2, "chance": 3}
    elif dataset == "Caltech-5V":
        params = {'dataname': "Caltech-5V", "feature_dim": 500,
                  "high_feature_dim": 800, "mid_dim": 512,
                  "layers1": 6, "layers2": 2, "chance": 1}
    elif dataset == "MNIST-USPS":
        params = {'dataname': "MNIST-USPS", "feature_dim": 128,
                  "high_feature_dim": 1000, "mid_dim": 1024,
                  "layers1": 4, "layers2": 2, "chance": 1}
    elif dataset == "msrcv1":
        params = {'dataname': "msrcv1", "feature_dim": 1024,
                  "high_feature_dim": 1200, "mid_dim": 1024,
                  "layers1": 2, "layers2": 4, "chance": 3}
    elif dataset == "scene":
        params = {'dataname': "scene", "feature_dim": 256,
                  "high_feature_dim": 256, "mid_dim": 512,
                  "layers1": 2, "layers2": 2, "chance": 1}
    elif dataset == "handwritten":
        params = {'dataname': "handwritten", "feature_dim": 1024,
                  "high_feature_dim": 256, "mid_dim": 768,
                  "layers1": 3, "layers2": 5, "chance": 1}
    else:
        raise NotImplementedError
    return params

def load_train(dataset):
    if dataset == "Caltech-2V":
        params = {'dataname': "Caltech-2V", "learning_rate": 0.0004, "temperature_f": 0.2, "temperature_l": 1.5,
                  "batch_size": 128, "mse_epochs": 130, "con_epochs": 30, "tune_epochs": 140, "feature_dim": 500,
                  "high_feature_dim": 1024, "mid_dim": 512,
                  "layers1": 2, "layers2": 6, "chance": 1, "seed": 180}
    elif dataset == "Caltech-3V":
        params = {'dataname': "Caltech-3V", "learning_rate": 0.0002, "temperature_f": 0.8, "temperature_l": 7,
                  "batch_size": 128, "mse_epochs": 190, "con_epochs": 70, "tune_epochs": 210, "feature_dim": 1024,
                  "high_feature_dim": 512, "mid_dim": 1024,
                  "layers1": 3, "layers2": 3, "chance": 1, "seed": 73}
    elif dataset == "Caltech-4V":
        params = {'dataname': "Caltech-4V", "learning_rate": 0.00025, "temperature_f": 1, "temperature_l": 2,
                  "batch_size": 100, "mse_epochs": 150, "con_epochs": 80, "tune_epochs": 130, "feature_dim": 512,
                  "high_feature_dim": 2048, "mid_dim": 512,
                  "layers1": 2, "layers2": 2, "chance": 3, "seed": 95}
    elif dataset == "Caltech-5V":
        params = {'dataname': "Caltech-5V", "learning_rate": 0.00015, "temperature_f": 1, "temperature_l": 6,
                  "batch_size": 100, "mse_epochs": 200, "con_epochs": 80, "tune_epochs": 200, "feature_dim": 500,
                  "high_feature_dim": 800, "mid_dim": 512,
                  "layers1": 6, "layers2": 2, "chance": 1, "seed": 2500}
    elif dataset == "MNIST-USPS":
        params = {'dataname': "MNIST-USPS", "learning_rate": 0.00025, "temperature_f": 0.7, "temperature_l": 8,
                  "batch_size": 100, "mse_epochs": 160, "con_epochs": 50, "tune_epochs": 200, "feature_dim": 128,
                  "high_feature_dim": 1000, "mid_dim": 1024,
                  "layers1": 4, "layers2": 2, "chance": 1, "seed": 260}
    elif dataset == "msrcv1":
        params = {'dataname': "msrcv1", "learning_rate": 0.0003, "temperature_f": 1, "temperature_l": 2.5,
                  "batch_size": 30, "mse_epochs": 200, "con_epochs": 40, "tune_epochs": 130, "feature_dim": 1024,
                  "high_feature_dim": 1200, "mid_dim": 1024,
                  "layers1": 2, "layers2": 4, "chance": 3, "seed": 31}
    elif dataset == "scene":
        params = {'dataname': "scene", "learning_rate": 0.00012, "temperature_f": 0.1, "temperature_l": 3.5,
                  "batch_size": 300, "mse_epochs": 190, "con_epochs": 40, "tune_epochs": 190, "feature_dim": 1000,
                  "high_feature_dim": 500, "mid_dim": 1024,
                  "layers1": 3, "layers2": 4, "chance": 3, "seed": 26}
    elif dataset == "handwritten":
        params = {'dataname': "handwritten", "learning_rate": 0.00035, "temperature_f": 1, "temperature_l": 2,
                  "batch_size": 200, "mse_epochs": 190, "con_epochs": 25, "tune_epochs": 110, "feature_dim": 1024,
                  "high_feature_dim": 256, "mid_dim": 768,
                  "layers1": 3, "layers2": 5, "chance": 1, "seed": 66}
    else:
        raise NotImplementedError
    return params

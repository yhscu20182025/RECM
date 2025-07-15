from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

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

class Flickr(Dataset):
    def __init__(self, path):
        self.x1 = scipy.io.loadmat(path+'MIRFlickr.mat')['X1'].astype(np.float32)
        self.x2 = scipy.io.loadmat(path+'MIRFlickr.mat')['X2'].astype(np.float32)
        self.y = scipy.io.loadmat(path+'MIRFlickr.mat')['Y']

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class NUSWIDE(Dataset):
    def __init__(self, path):
        self.x1 = scipy.io.loadmat(path+'NUSWIDE.mat')['X1'].astype(np.float32)
        self.x2 = scipy.io.loadmat(path+'NUSWIDE.mat')['X2'].astype(np.float32)
        self.y = scipy.io.loadmat(path+'NUSWIDE.mat')['Y']

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Caltech101(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Caltech101.mat')['Y'].astype(np.int32).reshape(9144, )
        self.V1 = scipy.io.loadmat(path + 'Caltech101.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Caltech101.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Caltech101.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'Caltech101.mat')['X4'].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'Caltech101.mat')['X5'].astype(np.float32)
        self.V6 = scipy.io.loadmat(path + 'Caltech101.mat')['X6'].astype(np.float32)

    def __len__(self):
        return 9144

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

def load_test(dataset):
    if dataset == "Caltech-2V":
        params = {'dataname': "Caltech-2V", "feature_dim": 768,
                  "high_feature_dim": 800, "mid_dim": 768,
                  "layers1": 2, "layers2": 4, "chance": 3,"neTwork":1}
    elif dataset == "Caltech-3V":
        params = {'dataname': "Caltech-3V", "feature_dim": 512,
                  "high_feature_dim": 1024, "mid_dim": 512,
                  "layers1": 2, "layers2": 4, "chance": 1,"neTwork":0}
    elif dataset == "Caltech-4V":
        params = {'dataname': "Caltech-4V", "feature_dim": 512,
                  "high_feature_dim": 800, "mid_dim": 1024,
                  "layers1": 3, "layers2": 5, "chance": 3,"neTwork":0}
    elif dataset == "Caltech-5V":
        params = {'dataname': "Caltech-5V", "feature_dim": 768,
                  "high_feature_dim": 1000, "mid_dim": 1024,
                  "layers1": 2, "layers2": 5, "chance": 2,"neTwork":0}
    elif dataset == "msrcv1":
        params = {'dataname': "msrcv1", "feature_dim": 1000,
                  "high_feature_dim": 800, "mid_dim": 768,
                  "layers1": 2, "layers2": 6, "chance": 3,"neTwork":0}
    elif dataset == "scene":
        params = {'dataname': "scene", "feature_dim": 512,
                  "high_feature_dim": 1024, "mid_dim": 512,
                  "layers1": 2, "layers2": 4, "chance": 3,"neTwork":0}
    elif dataset == "handwritten":
        params = {'dataname': "handwritten", "feature_dim": 1000,
                  "high_feature_dim": 1000, "mid_dim": 1024,
                  "layers1": 3, "layers2": 6, "chance": 3,"neTwork":0}
    elif dataset == "Caltech101":
        params = {'dataname': "Caltech101", "feature_dim": 768,
                  "high_feature_dim": 768, "mid_dim": 500,
                  "layers1": 1, "layers2": 4, "chance": 1,"neTwork":0}
    elif dataset == "Flickr":
        params = {'dataname': "Flickr", "feature_dim": 512,
                  "high_feature_dim": 500, "mid_dim": 500,
                  "layers1": 2, "layers2": 2, "chance": 3,"neTwork":0}
    elif dataset == "NUSWIDE":
        params = {'dataname': "NUSWIDE", "feature_dim": 512,
                  "high_feature_dim": 500, "mid_dim": 512,
                  "layers1": 1, "layers2": 2, "chance": 1,"neTwork":1}
    else:
        raise NotImplementedError
    return params

def load_train(dataset):
    if dataset == "Caltech-2V":
        params = {'dataname': "Caltech-2V","batch_size": 300, "learning_rate": 0.0005, "temperature_f": 1, "temperature_l": 12,
                 "con_epochs": 40,  "feature_dim": 768, "high_feature_dim": 800, "mid_dim": 768,
                 "layers1": 2, "layers2": 4, "chance": 3, "seed": 280,"miss_rate":0,"neTwork":1}
    elif dataset == "Caltech-3V":
        params = {'dataname': "Caltech-3V","batch_size": 128, "learning_rate": 0.0003, "temperature_f": 1, "temperature_l": 7,
                 "con_epochs": 30,  "feature_dim": 768, "high_feature_dim": 800, "mid_dim": 1024,
                 "layers1": 2, "layers2": 4, "chance": 1, "seed": 28,"miss_rate":0,"neTwork":0}
    elif dataset == "Caltech-4V":
        params = {'dataname': "Caltech-4V","batch_size": 128, "learning_rate": 0.0003, "temperature_f": 0.9, "temperature_l": 12,
                 "con_epochs": 35,  "feature_dim": 512, "high_feature_dim": 800, "mid_dim": 1024,
                 "layers1": 3, "layers2": 5, "chance": 3, "seed": 220,"miss_rate":0}
    elif dataset == "Caltech-5V":
        params = {'dataname': "Caltech-5V","batch_size": 200, "learning_rate": 0.00035, "temperature_f": 0.9, "temperature_l": 15,
                 "con_epochs": 20,  "feature_dim": 768, "high_feature_dim": 1000, "mid_dim": 1024,
                 "layers1": 2, "layers2": 5, "chance": 2, "seed": 14,"miss_rate":0,"neTwork":1}
    elif dataset == "Caltech101":
        params = {'dataname': "Caltech101", "learning_rate": 0.0005, "temperature_f": 0.8, "temperature_l": 9,
                  "batch_size": 128, "con_epochs": 10, "feature_dim": 768,
                  "high_feature_dim": 768, "mid_dim": 500,
                  "layers1": 1, "layers2": 4, "chance": 1, "seed": 26,"miss_rate":0,"neTwork":0}
    elif dataset == "msrcv1":
        params = {'dataname': "msrcv1", "learning_rate": 0.0004, "temperature_f": 1, "temperature_l": 6,
                  "batch_size": 40,  "con_epochs": 40,  "feature_dim": 1000,
                  "high_feature_dim": 800, "mid_dim": 768,
                  "layers1": 2, "layers2": 6, "chance": 3, "seed": 79,"miss_rate":0}
    elif dataset == "scene":
        params = {'dataname': "scene", "learning_rate": 0.0004, "temperature_f": 1, "temperature_l": 13,
                  "batch_size": 200, "con_epochs": 20, "feature_dim": 512,
                  "high_feature_dim": 1024, "mid_dim": 512,
                  "layers1": 2, "layers2": 4, "chance": 3, "seed": 200,"miss_rate":0,"neTwork":0}
    elif dataset == "handwritten":
        params = {'dataname': "handwritten", "learning_rate": 0.00035, "temperature_f": 0.8, "temperature_l": 13,
                  "batch_size": 300, "con_epochs": 20, "feature_dim": 1000,
                  "high_feature_dim": 1000, "mid_dim": 1024,
                  "layers1": 3, "layers2": 6, "chance": 3, "seed": 25,"miss_rate":0,"neTwork":0}
    elif dataset == "Flickr":
        params = {'dataname': "Flickr", "learning_rate": 0.0004, "temperature_f": 0.7, "temperature_l": 15,
                  "batch_size": 500, "con_epochs": 10, "feature_dim": 512,
                  "high_feature_dim": 500, "mid_dim": 500,
                  "layers1": 2, "layers2": 2, "chance": 3, "seed": 10,"miss_rate":0,"neTwork":0}
    elif dataset == "NUSWIDE":
        params = {'dataname': "NUSWIDE", "learning_rate": 0.0003, "temperature_f": 0.7, "temperature_l": 11,
                  "batch_size": 400,  "feature_dim": 512,
                  "high_feature_dim": 800, "mid_dim": 512,"con_epochs": 5,
                  "layers1": 1, "layers2": 2, "chance": 2, "seed": 55,"miss_rate":0,"neTwork":1}

    else:
        raise NotImplementedError
    return params


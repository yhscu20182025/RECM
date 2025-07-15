import torch
from networks import Network
from metric import valid,valid4,valid2
from Dataprocessing import load_data, load_test
from train import get_mask

# msrcv1
# scene
# handwritten
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Caltech101
# IAPR
# NUSWIDE
dataname = "Fashion"
modelname = "RECM"
params = load_test(dataname)
args = params
Dataname = args["dataname"]
dataset = Dataname
feature_dim = args["feature_dim"]
high_feature_dim = args["high_feature_dim"]
mid_dim = args["mid_dim"]
layers1 = args["layers1"]
layers2 = args["layers2"]
chance = args["chance"]
neTwork= args["neTwork"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(Dataname)
model = Network(view, dims, feature_dim, high_feature_dim, mid_dim, layers1, layers2, chance, class_num, device, neTwork)
model = model.to(device)

# 仅加载最终训练好的模型权重
checkpoint = torch.load(f'../../Models/RECM/{Dataname}.pth', map_location=device)
model.load_state_dict(checkpoint)
mask = get_mask(view, data_size)
print("dataname:"+dataname)
valid(model, device, dataset, view, data_size, class_num, mask)

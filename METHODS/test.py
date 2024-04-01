import torch
from network import Network
from metric import valid
from dataprocessing import load_data,load_test

# MNIST-USPS
# scene
# msrcv1
# hanwritten
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
dataname="Caltech-2V"
params = load_test(dataname)
args=params
Dataname = args["dataname"]

dataset = Dataname

feature_dim = args["feature_dim"]
high_feature_dim = args["high_feature_dim"]
mid_dim = args["mid_dim"]
encoderlayers=args["layers1"]
decoderlayers=args["layers2"]
mlpandsoftmax=args["chance"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(Dataname)
model = Network(view, dims, feature_dim, high_feature_dim, mid_dim,encoderlayers,decoderlayers, mlpandsoftmax,class_num, device)
model = model.to(device)
checkpoint = torch.load('../MODELS/'+ Dataname + '.pth', map_location=lambda storage, loc: storage.cuda(0))
model.load_state_dict(checkpoint)

print("Dataset:{}".format(Dataname))
print("Datasize:" + str(data_size))
#print("Loading models...")
valid(model, device, dataset, view, data_size, class_num, eval_h=False)

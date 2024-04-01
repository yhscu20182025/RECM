import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import random
from loss import Loss
from dataprocessing import load_data,load_train

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    Dataname = args["dataname"]
    learning_rate = args["learning_rate"]
    batch_size = args["batch_size"]
    dataset = Dataname
    temperature_f = args["temperature_f"]
    temperature_l = args["temperature_l"]
    weight_decay = 0.
    mse_epochs = args["mse_epochs"]
    con_epochs = args["con_epochs"]
    tune_epochs = args["tune_epochs"]
    feature_dim = args["feature_dim"]
    high_feature_dim = args["high_feature_dim"]
    mid_dim = args["mid_dim"]
    encoderlayers = args["layers1"]
    decoderlayers = args["layers2"]
    mlpandsoftmax = args["chance"]
    seed = args["seed"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    dataset, dims, view, data_size, class_num = load_data(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    def pretrain(epoch):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            xrs, _ = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    def contrastive_train(epoch):
        tot_loss = 0.
        mes = torch.nn.MSELoss()
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, pres, qs, xrs, zs = model.forward_plot1(xs)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_label2(pres[v], qs[w]))
                    loss_list.append(criterion.forward_label2(pres[w], qs[v]))
                loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(criterion.forward_label2(pres[v], qs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

    def contrastive_train2(epoch):
        tot_loss = 0.
        mes = torch.nn.MSELoss()
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, qs, xrs, zs = model.forward2(xs)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_label(qs[v], qs[w]))
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

    T = 1
    # print(Dataname)
    for i in range(T):


        model = Network(view, dims, feature_dim, high_feature_dim, mid_dim,encoderlayers,decoderlayers, mlpandsoftmax,class_num, device)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = Loss(batch_size, class_num, temperature_f, temperature_l, device).to(device)

        epoch = 1
        while epoch <= mse_epochs:
            pretrain(epoch)
            epoch += 1
        while epoch <= mse_epochs + con_epochs:
            contrastive_train(epoch)

            epoch += 1

        while epoch <= mse_epochs + con_epochs + tune_epochs:
            contrastive_train2(epoch)
            if epoch == mse_epochs + con_epochs + tune_epochs:


                torch.save(model.state_dict(),Dataname + '.pth')
                acc, nmi, pur = valid(model, device, dataset, view, data_size, batch_size, eval_h=False)
            epoch += 1

# MNIST-USPS
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# handwritten
# msrcv1
# scene
if __name__ == '__main__':

    torch.set_num_threads(2)
    dataname="Caltech-2V"
    params = load_train(dataname)
    main(params)

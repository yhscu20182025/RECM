import os
import torch
from networks import Network
from metric import valid2 ,get_mask,valid3
from torch.utils.data import Dataset
import numpy as np
import random
from loss import Loss
from Dataprocessing import load_data, load_train
import pandas as pd

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min - self.delta:
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
    con_epochs = args["con_epochs"]
    mse_epochs = 200
    tune_epochs = 100
    feature_dim = args["feature_dim"]
    high_feature_dim = args["high_feature_dim"]
    mid_dim = args["mid_dim"]
    layers1 = args["layers1"]
    layers2 = args["layers2"]
    chance = args["chance"]
    neTwork=args["neTwork"]
    seed = args["seed"]
    patience = 15
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
        for batch_idx, (xs, _, indices) in enumerate(data_loader):
            #xs = apply_mask(xs, mask, indices, view)
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            xrs, _ = model(xs)
            loss_list = [criterion(xs[v], xrs[v]) for v in range(view)]
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        print(f'Epoch {epoch}, Loss: {tot_loss / len(data_loader):.6f}')
        return tot_loss / len(data_loader)

    def contrastive_train(epoch):
        tot_loss = 0.
        for batch_idx, (xs, _, indices) in enumerate(data_loader):
            #xs = apply_mask(xs, mask, indices, view)
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, pres, qs, xrs, zs = model.forward_plot1(xs)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_label2(pres[v], qs[w]))
                    loss_list.append(criterion.forward_label2(pres[w], qs[v]))
                # loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(criterion.forward_label2(pres[v], qs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        print(f'Epoch {epoch}, Loss: {tot_loss / len(data_loader):.6f}')
        return tot_loss / len(data_loader)

    def contrastive_train2(epoch):
        tot_loss = 0.
        # mes = torch.nn.MSELoss()
        for batch_idx, (xs, _, indices) in enumerate(data_loader):
            # xs = apply_mask(xs, mask, indices, view)  # Apply mask to xs
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, qs, xrs, zs = model.forward2(xs)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_label(qs[v], qs[w]))
                # loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        return tot_loss / len(data_loader)

    T = 1
    for i in range(T):
        model = Network(view, dims, feature_dim, high_feature_dim, mid_dim, layers1, layers2, chance, class_num, device,neTwork)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = Loss(batch_size, class_num, temperature_f, temperature_l, device).to(device)
        # criterion = Loss(batch_size, class_num, temperature_f, 1, device).to(device)
        early_stopping = EarlyStopping(patience=patience, path=Dataname + 'train.pth')

        epoch = 1
        while epoch <= mse_epochs:
            val_loss = pretrain(epoch)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping during pretraining epoch{}".format(epoch))
                break
            epoch += 1

        epoch = 1
        while epoch <= con_epochs:
            val_loss = contrastive_train(epoch)
            epoch += 1

        epoch = 1
        best_acc, best_nmi, best_ari = 0, 0, 0
        # early_stopping = EarlyStopping(patience=patience, path=Dataname + 'train_contrastive2.pth')
        while epoch <= tune_epochs:
            val_loss = contrastive_train2(epoch)
            if epoch>1:
                acc, nmi, ari = valid2(model, device, dataset, view, data_size, class_num)

                if acc > best_acc:
                    best_acc, best_nmi, best_ari = acc, nmi, ari
                    state = model.state_dict()
                    torch.save(model.state_dict(), f'../../Models/RECM/{dataname}TRAIN.pth')
            epoch += 1
        acc,nmi,ari= valid2(model, device, dataset, view, data_size, batch_size)
        print(
            'The best clustering performace: ACC = {:.4f} NMI = {:.4f} ARI={:.4f}'.format(best_acc, best_nmi, best_ari))
        checkpoint_files = [
            Dataname + 'train.pth',
            Dataname + 'train_contrastive2.pth'
        ]
        for file in checkpoint_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        return acc



if __name__ == '__main__':
    torch.set_num_threads(4)
    # msrcv1
    # scene
    # handwritten
    # Caltech-2V
    # Caltech-3V
    # Caltech-4V
    # Caltech-5V
    dataname="Fashion"
    params = load_train(dataname)
    main(params)

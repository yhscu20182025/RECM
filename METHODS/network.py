import math

import torch.nn as nn
from layers import Encoder,Decoder,Encoder1
#nnitrain layers2
from scipy.optimize import linear_sum_assignment
import torch


def kmeans(X, num_clusters, num_iterations=100, tol=1e-4, device="cuda"):
    """Improved K-Means with K-Means++ init and early stopping."""
    n_samples = X.shape[0]

    # 1. K-Means++ 初始化
    centroids = X[torch.randint(0, n_samples, (1,), device=device)]  # 随机选第一个中心点
    for _ in range(num_clusters - 1):
        distances = torch.cdist(X, centroids).min(dim=1).values  # 每个点到最近中心的距离
        prob = distances / distances.sum()
        new_center_idx = torch.multinomial(prob, 1)
        centroids = torch.cat([centroids, X[new_center_idx]])

    # 2. 迭代优化
    for _ in range(num_iterations):
        # 计算距离并分配簇
        distances = torch.cdist(X, centroids)
        cluster_indices = torch.argmin(distances, dim=1)

        # 更新中心点
        new_centroids = torch.zeros_like(centroids)
        for i in range(num_clusters):
            mask = cluster_indices == i
            if torch.any(mask):
                new_centroids[i] = X[mask].mean(dim=0)
            else:
                new_centroids[i] = X[torch.randint(0, n_samples, (1,), device=device)]  # 随机选一个新点

        # 早停检查：如果中心点变化很小，提前终止
        shift = torch.norm(new_centroids - centroids)
        if shift < tol:
            break
        centroids = new_centroids

    return centroids, cluster_indices



class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim,mid_dim,layers1,layers2,chance, class_num, device,neTwork=1):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.classnum=class_num
        self.highfeaturedim=high_feature_dim
        self.device=device
        if neTwork==0:
            for v in range(view):
                self.encoders.append(Encoder(input_size[v], feature_dim,mid_dim,layers1).to(device))
                self.decoders.append(Decoder(input_size[v], feature_dim,mid_dim,layers2).to(device))
        elif neTwork==1:
            for v in range(view):
                self.encoders.append(Encoder1(input_size[v], feature_dim, mid_dim, layers1).to(device))
                self.decoders.append(Decoder(input_size[v], feature_dim, mid_dim, layers2).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)


        if chance == 1:
            self.label_contrastive_module2 = nn.Sequential(
                nn.Linear(feature_dim, high_feature_dim),
                nn.ReLU(),
                nn.Linear(high_feature_dim, class_num),
                nn.Softmax(dim=1)
            )
        elif chance == 2:
            self.label_contrastive_module2 = nn.Sequential(
                nn.Linear(feature_dim, high_feature_dim),
                nn.Linear(high_feature_dim, class_num),
                nn.Softmax(dim=1)
            )
        elif chance == 3:
            self.label_contrastive_module2 = nn.Sequential(
                nn.Linear(feature_dim, class_num),
                nn.Softmax(dim=1)
            )
        self.view = view

        self.query = nn.Linear(class_num, class_num)
        self.key = nn.Linear(class_num, class_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xs):
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return xrs, zs

    def forward2(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            l = self.label_contrastive_module2(z)
            xr = self.decoders[v](z)
            zs.append(z)
            qs.append(l)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot1(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        pres= []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.label_contrastive_module2(z)
            predicted_labels = torch.argmax(h, dim=1)

            unique_labels = torch.unique(predicted_labels)
            k = unique_labels.size(0)

            Q = self.query(h)
            K = self.key(h)

            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
            attention_weights = self.softmax(attention_scores)

            # w = torch.matmul(h, h.T)
            w = attention_weights

            d = torch.sum(w, dim=1)
            d_sqrt_inv = torch.sqrt(1.0 / (d + 1e-8))
            l = torch.diag(d_sqrt_inv.float()) @ w.float() @ torch.diag(d_sqrt_inv.float())
            l[torch.isnan(l)] = 0.0
            l[torch.isinf(l)] = 0.0

            eigenvalues, eigenvectors = torch.linalg.eig(l)
            real_parts = eigenvalues.real

            sorted_indices = torch.argsort(real_parts)
            sorted_indices = sorted_indices
            k_eigenvectors = eigenvectors[:, sorted_indices[0:self.classnum]].real


            centroids, labels = kmeans(k_eigenvectors, k,100)
            similarity_matrix = torch.zeros((torch.max(predicted_labels) + 1, torch.max(labels) + 1))
            similarity_matrix = similarity_matrix.to(self.device)

            for i, j in zip(predicted_labels, labels):
                similarity_matrix[i, j] += 1


            similarity_matrix_cpu = similarity_matrix.cpu()
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix_cpu)
            y_pred_mapped = torch.clone(labels)
            for i, j in zip(col_ind, row_ind):
                y_pred_mapped[labels == i] = j


            num_labels = self.classnum

            one_hot_labels = torch.zeros((labels.size(0), num_labels), device=labels.device)
            one_hot_labels.scatter_(1, y_pred_mapped.unsqueeze(1), 1)

            xr = self.decoders[v](z)

            hs.append(h)
            pres.append(h)
            zs.append(z)
            qs.append(one_hot_labels)

            xrs.append(xr)
        return hs,pres, qs, xrs, zs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module2(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds




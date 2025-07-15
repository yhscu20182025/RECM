from scipy.io import loadmat
from sklearn.metrics import accuracy_score, v_measure_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import math

def get_mask(view_num, data_size, missing_ratio):
    """
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size * missing_ratio)
    data_ind = list(range(data_size))
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones((data_size, view_num))

    for j in miss_ind:
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            rand_v[observed_ind] = 1
            rand_v[~observed_ind] = 0
            if 0 < np.sum(rand_v) < view_num:
                break
        mask[j] = rand_v

    return mask

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(label, pred):
    y_true=label
    y_pred=pred
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    acc=sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    values = np.array([acc, nmi, ari])
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    pur = accuracy_score(y_true, y_voted_labels)
    return acc,nmi,ari



def apply_mask(xs, mask, indices, view):
    # Apply mask to the batch
    for v in range(view):
        for i, idx in enumerate(indices):
            if mask[idx, v] == 0:
                xs[v][i] = 0  # Set missing data to zero or another indicator
    return xs

def inference3(loader, model, device, view, data_size,class_num,mask):
    model.eval()
    features = []  # 存储融合后的特征（用于K-means）
    labels_vector = []

    for step, (xs, y, indices) in enumerate(loader):
        xs = apply_mask(xs, mask, indices, view)  # Apply mask to xs
        for v in range(view):
            xs[v] = xs[v].to(device)

        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            # 假设你的模型能返回特征（如zs或H），替换下面的zs为实际特征
            # 如果只有qs，可以用qs的平均作为特征（但效果可能不如低层特征）
            zs = [q.detach() for q in qs]  # 示例：用qs作为替代特征
            fused_feature = torch.cat(zs, dim=1).mean(dim=1, keepdim=True)  # 简单融合

        features.append(fused_feature.cpu().numpy())
        labels_vector.append(y.numpy())

    # 合并所有batch数据
    features = np.concatenate(features, axis=0).reshape(data_size, -1)
    labels_vector = np.concatenate(labels_vector).reshape(data_size)

    # 使用K-means聚类
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    total_pred = kmeans.fit_predict(features)

    return total_pred, labels_vector

def inference(loader, model, device, view, data_size, mask):
    model.eval()
    soft_vector = []
    labels_vector = []

    for step, (xs, y, indices) in enumerate(loader):
        xs = apply_mask(xs, mask, indices, view)  # Apply mask to xs
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            q = sum(qs)/view
        q = q.detach()
        soft_vector.extend(q.cpu().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, labels_vector



def valid(model, device, dataset, view, data_size, batch_size, mask):
    test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    total_pred, labels_vector = inference(test_loader, model, device, view, data_size, mask)
    acc,nmi,ari= evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))
    return acc,nmi,ari

# def inference(loader, model, device, view, data_size, mask):
#     model.eval()
#     # 为每个视图存储预测结果和特征
#     view_predictions = [[] for _ in range(view)]
#     labels_vector = []
#
#     for step, (xs, y, indices) in enumerate(loader):
#         xs = apply_mask(xs, mask, indices, view)  # Apply mask to xs
#         for v in range(view):
#             xs[v] = xs[v].to(device)
#         with torch.no_grad():
#             qs, preds = model.forward_cluster(xs)
#
#         # 存储每个视图的预测结果
#         for v in range(view):
#             view_predictions[v].extend(qs[v].cpu().numpy())
#         labels_vector.extend(y.numpy())
#
#     labels_vector = np.array(labels_vector).reshape(data_size)
#
#     # 评估每个视图的表现
#     best_acc = -1
#     best_pred = None
#     for v in range(view):
#         current_pred = np.argmax(np.array(view_predictions[v]), axis=1)
#         acc, nmi, ari = evaluate(labels_vector, current_pred)
#         if acc > best_acc:
#             best_acc = acc
#             best_pred = current_pred
#
#     return best_pred, labels_vector
#
#
# def valid(model, device, dataset, view, data_size, batch_size, mask):
#     test_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#     )
#
#     total_pred, labels_vector = inference(test_loader, model, device, view, data_size, mask)
#     acc, nmi, ari = evaluate(labels_vector, total_pred)
#     print('Best View - ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))
#     return acc, nmi, ari

def inference2(loader, model, device, view, data_size):
    model.eval()
    soft_vector = []
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        #xs = apply_mask(xs, mask, indices, view)  # Apply mask to xs
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            q = sum(qs)/view
        q = q.detach()
        soft_vector.extend(q.cpu().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, labels_vector



def valid2(model, device, dataset, view, data_size, batch_size):
    test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    total_pred, labels_vector = inference2(test_loader, model, device, view, data_size)
    acc,nmi,ari= evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))
    return acc,nmi,ari

# def inference2(loader, model, device, view, data_size):
#     model.eval()
#     # 为每个视图存储预测结果和特征
#     view_predictions = [[] for _ in range(view)]
#     labels_vector = []
#
#     for step, (xs, y, indices) in enumerate(loader):
#         for v in range(view):
#             xs[v] = xs[v].to(device)
#         with torch.no_grad():
#             qs, preds = model.forward_cluster(xs)
#
#         # 存储每个视图的预测结果
#         for v in range(view):
#             view_predictions[v].extend(qs[v].cpu().numpy())
#         labels_vector.extend(y.numpy())
#
#     labels_vector = np.array(labels_vector).reshape(data_size)
#
#     # 评估每个视图的表现
#     best_acc = -1
#     best_pred = None
#     for v in range(view):
#         current_pred = np.argmax(np.array(view_predictions[v]), axis=1)
#         acc, nmi, ari = evaluate(labels_vector, current_pred)
#         if acc > best_acc:
#             best_acc = acc
#             best_pred = current_pred
#
#     return best_pred, labels_vector
#
#
# def valid2(model, device, dataset, view, data_size, batch_size):
#     test_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#     )
#
#     total_pred, labels_vector = inference2(test_loader, model, device, view, data_size)
#     acc, nmi, ari = evaluate(labels_vector, total_pred)
#     print('Best View - ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))
#     return acc, nmi, ari

def inference3(loader, model, device, view, data_size):
    model.eval()
    soft_vector = []
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        with torch.no_grad():
            qs, preds, weights = model.forward_cluster(xs)
            q_stack = torch.stack(qs, dim=1)  # [batch, view, class]
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, view, 1]
            q = torch.sum(q_stack * weights, dim=1)  # 加权求和

        soft_vector.extend(q.cpu().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    return total_pred, labels_vector



def valid3(model, device, dataset, view, data_size, batch_size):
    test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    total_pred, labels_vector = inference3(test_loader, model, device, view, data_size)
    acc,nmi,ari= evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))
    return acc,nmi,ari

def inference4(loader, model, device, view, data_size,mask):
    model.eval()
    soft_vector = []
    labels_vector = []

    for step, (xs, y, indices) in enumerate(loader):
        xs = apply_mask(xs, mask, indices, view)  # Apply mask to xs
        for v in range(view):
            xs[v] = xs[v].to(device)

        with torch.no_grad():
            qs, preds, weights = model.forward_cluster(xs)
            q_stack = torch.stack(qs, dim=1)  # [batch, view, class]
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, view, 1]
            q = torch.sum(q_stack * weights, dim=1)  # 加权求和

        soft_vector.extend(q.cpu().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    return total_pred, labels_vector



def valid4(model, device, dataset, view, data_size, batch_size,mask):
    test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    total_pred, labels_vector = inference4(test_loader, model, device, view, data_size,mask)
    acc,nmi,ari= evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))
    return acc,nmi,ari


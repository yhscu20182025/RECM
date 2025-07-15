import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
eps=1e-8

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def mask_correlated_samples2(self, N):
        m1 = torch.ones((N//2, N//2))
        m1 = m1.fill_diagonal_(0)
        m2 = torch.zeros((N//2, N//2))
        mask1 = torch.cat([m1, m2], dim=1)
        mask2 = torch.cat([m2, m1], dim=1)
        mask = torch.cat([mask1, mask2], dim=0)
        mask = mask.bool()
        return mask

    def mask_correlated_samples3(self, N):
        m1 = torch.ones((N//2, N//2))
        m1 = m1.fill_diagonal_(0)
        m2 = torch.zeros((N//2, N//2))
        mask1 = torch.cat([m2, m1], dim=1)
        mask2 = torch.cat([m1, m2], dim=1)
        mask = torch.cat([mask1, mask2], dim=0)
        mask = mask.bool()
        return mask


    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i=torch.clamp(p_i,min=eps)
        ne_i =  (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        p_j=torch.clamp(p_j,min=eps)
        ne_j =  (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_f
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        entropy/=N
        return loss+entropy

    def forward_label2(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i = torch.clamp(p_i, min=eps)
        ne_i = (p_i * torch.log(p_i)).sum()

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = torch.zeros(N,1).to(positive_clusters.device)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        ne_i/=(N/2)
        return loss+ne_i

    def forward_feature2(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)


        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)


        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_feature(self, h_i, h_j,r=3.0):
        z1=h_i
        z2=h_j
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.sum(z1 * z2, dim=1, keepdim=True) / z1.shape[0]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.sum(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1), dim=1,
                               keepdim=True) \
                     / (z1.shape[0] * (z1.shape[0] - 1))

        return loss_part1 + loss_part2
import torch
from torch import nn
import torch.nn.functional as F

# Nicholas Lower
# Triplet loss function for GaitPart model pipeline.
# Original paper - doi: 10.1109/CVPR42600.2020.01423 

class TripletLoss(nn.Module):
    def __init__(self, margin = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embs, labels):
        embs = embs.permute(1, 0, 2).contiguous().float()

        ref_embed, ref_label = embs, labels
        dist = self.ComputeDistance(embs, ref_embed)
        p_dist, n_dist = self.ToTriplets(labels, ref_label, dist)
        loss = F.relu((p_dist - n_dist) + self.margin)

        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        return loss_avg

    def AvgNonZeroReducer(self, loss):
        sum = loss.sum(-1)
        num = (loss != 0).sum(-1).float()

        avg = sum / (num + 1.0e-9) # For stability
        avg[num == 0] = 0

        return avg, num

    def ComputeDistance(self, x, y):
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)
        inner = x.matmul(y.transpose(-1, -2))
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))

        return dist

    def ToTriplets(self, row_labels, col_labels, dist):
        matches = (row_labels.unsqueeze(1) == col_labels.unsqueeze(0)).byte()
        diff = matches ^ 1
        mask = matches.unsqueeze(2) * diff.unsqueeze(1)
        anchor_idx, pos_idx, neg_idx = torch.where(mask)

        p_dist = dist[:, anchor_idx, pos_idx]
        n_dist = dist[:, anchor_idx, neg_idx]

        return p_dist, n_dist
import torch
from torch import nn
import torch.functional as F

import numpy as np
import copy as cp

# Nicholas Lower
# CS 722 HW4

class GaitPart(nn.Module):
    def __init__(self):
        super().__init__()

        self.FPFE = BlockWrapper(FPFE())
        self.HPP = BlockWrapper(HPP())
        self.TFA = TFA(in_channels=128, parts_num=128)

        self.Head = Head()

    def forward(self, inputs):

        # Input is [b, seq, ch, h, w]
        # b - Batch size, seq - sequence length, ch - channels, h - height, w - width
        out = self.FPFE(inputs)
        out = self.HPP(out)
        out = self.TFA(out)

        out = self.Head(out.permute(1, 0, 2).contiguous())
        embs = out.permute(1, 0, 2).contiguous()

        return embs


class FPFE(nn.Module):
    '''
        Defines Frame-level Part Feature Extractor (FPFE) module in GaitPart.
        First module of whole network, takes in frame and extracts part-based features.

        Uses series of basic 2D convolutional layers, focal convolutional layers, and max pooling to extract
        features relating to the various body parts of a human subject.

        Returns a series of feature maps, each consisting of some feature of some body part (same parts are grouped together
        along 1st dimension).
    '''
    def __init__(self):
        super().__init__()
        self.begin_convs = nn.Sequential(
            nn.Conv2d(1, 32, 5,  padding=2, bias=False), nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle_convs = nn.Sequential(
            FocalConv(32, 64, 3, padding=1, halving=2), nn.LeakyReLU(inplace=True),
            FocalConv(64, 64, 3, padding=1, halving=2), nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.last_convs = nn.Sequential(
            FocalConv(64, 128, 3, padding=1, halving=3), nn.LeakyReLU(inplace=True),
            FocalConv(128, 128, 3, padding=1, halving=3), nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.begin_convs(x)
        x = self.middle_convs(x)
        return self.last_convs(x)


class FocalConv(nn.Module):
    '''
        Special convolution layer that segments image into parts and convolves on each part in parallel.

        in_channels, out_channels, kernel_size, padding - Same meaning as basic Conv2D.
        halving - Number of parts to consider (2^halving parts)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding, halving):
        super().__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding= padding, bias=False )

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2 ** self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(t) for t in z], 2)

        return z


class HPP(nn.Module):
    '''
        Defines the Horizontal Pooling module in GaitPart.

        Used to separate the feature maps from FPFE into part-based temporal feature vectors for TFA by combining
        average pooling and max pooling.
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        n, ch = x.size()[:2]
        features = []
        z = x.view(n, ch, 16, -1)
        z = z.mean(-1) + z.max(-1)[0]
        features.append(z)
        return torch.cat(features, -1)


class TFA(nn.Module):
    '''
        Defines the Temporal Feature Aggregator module in GaitPart.

        Used to capture temporal features across frames given parts data. Utilizes inner Micro-motion Capture Modules (MCM)
        to extract "micro" movements in data as the smaller movements within a single gait cycle are more telling than
        whole periodic video sequence. Each part gets own parallel MCM. MCM is comprised of two Micro-motion Template Builders
        (MTB) and Temporal Pooling (TP).
    '''
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super().__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1 layers
        conv3x1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels, 1, bias=False))
        self.conv1d3x1 = clone(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB2 layers
        conv3x3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1, bias=False), nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels, 3, padding=1, bias=False))
        self.conv1d3x3 = clone(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling
        self.TP = torch.max

    def forward(self, x):
        b, seq, ch, parts = x.size()
        x = x.permute(3, 0, 2, 1).contiguous()
        feature = x.split(1, 0)
        x = x.view(-1, ch, seq)

        # MTB1
        logits = torch.cat([conv(f.squeeze(0)).unsqueeze(0) for conv, f in zip(self.conv1d3x1, feature)], 0)
        scores = torch.sigmoid(logits)

        micro_motion1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        micro_motion1 = micro_motion1.view(parts, b, ch, seq)
        micro_motion1 = micro_motion1 * scores

        # MTB2
        logits = torch.cat([conv(f.squeeze(0)).unsqueeze(0) for conv, f in zip(self.conv1d3x3, feature)], 0)
        scores= torch.sigmoid(logits)

        micro_motion2 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        micro_motion2 = micro_motion2.view(parts, b, ch, seq)
        micro_motion2 = micro_motion2 * scores

        # Temporal Pooling
        res = self.TP(micro_motion1 + micro_motion2, dim=-1)[0]
        res = res.permute(1, 0, 2).contiguous()

        return res


class BlockWrapper(nn.Module):
    '''
        Defines wrapper module in GaitPart that preps and restructures incoming and outgoing data for a block.
    '''
    def __init__(self, forward_block):
        super().__init__()
        self.forward_block = forward_block

    def forward(self, x):
        b, seq, ch, h, w = x.size()
        x = self.forward_block(x.view(-1, ch, h, w))
        _ = x.size()
        _ = [b, seq] + [*_[1:]]
        return x.view(*_)


class Head(nn.Module):
    '''
        Defines Head module in GaitPart. The final component of GaitPart, the head acts as output layer for the whole
        network, taking the part-segmented temporal features and producing a final embedding matrix (an embedding feature
        vector for each part).
    '''
    def __init__(self, parts_num = 16, in_channels = 128, out_channels = 128):
        super().__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(
                                         torch.zeros(parts_num, in_channels, out_channels)))

    def forward(self, x):
        out = x.matmul(self.fc_bin)
        return out


def clone(module, N):
    '''
        Utility function to clone N number of GaitPart modules.
    '''
    return nn.ModuleList([cp.deepcopy(module) for _ in range(N)])
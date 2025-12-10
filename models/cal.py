import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet as resnet

__all__ = ['WSDAN_CAL']
EPSILON = 1e-6


class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF_bpool = features * attentions[:, i:i + 1, ...]
                AiF = self.pool(AiF_bpool).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)

        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)
        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + EPSILON)
        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(num_features, num_features // reduction, bias=False)
        self.fc2 = nn.Linear(num_features // reduction, num_features, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)
        gmp = F.adaptive_max_pool2d(x, 1).view(B, C)
        gap_out = self.fc2(self.relu(self.fc1(gap)))
        gmp_out = self.fc2(self.relu(self.fc1(gmp)))
        channel_weights = self.sigmoid(gap_out + gmp_out)
        channel_weights = channel_weights.unsqueeze(2).unsqueeze(3)
        enhanced_features = channel_weights * x
        return enhanced_features, channel_weights.squeeze()


class WSDAN_CAL(nn.Module):
    def __init__(self, num_classes, M=32, net='resnet152', pretrained=False):
        super(WSDAN_CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        if 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s. Only ResNet backbones are supported.' % net)

        self.channel_attention = ChannelAttention(self.num_features, reduction=16)
        self.attentions = nn.Conv2d(self.num_features, self.M, kernel_size=1)
        self.bap = BAP(pool='GAP')
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

    def visualize(self, x):
        batch_size = x.size(0)
        feature_maps = self.features(x)
        enhanced_features, _ = self.channel_attention(feature_maps)
        attention_maps = self.attentions(enhanced_features)
        feature_matrix, _ = self.bap(enhanced_features, attention_maps)
        p = self.fc(feature_matrix * 100.)
        return p, attention_maps

    def forward(self, x, return_attention_weights=False):
        batch_size = x.size(0)
        feature_maps = self.features(x)
        enhanced_features, channel_weights = self.channel_attention(feature_maps)
        attention_maps = self.attentions(enhanced_features)
        feature_matrix, feature_matrix_hat = self.bap(enhanced_features, attention_maps)
        p_coarse = self.fc(feature_matrix * 100.)
        p_counterfactual = self.fc(feature_matrix_hat * 100.)
        p_aux = p_coarse - p_counterfactual

        if self.training:
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)

        if return_attention_weights:
            return p_coarse, p_aux, feature_matrix, attention_map, channel_weights, attention_maps
        else:
            return p_coarse, p_aux, feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}
        print("model_dict", len(pretrained_dict), len(state_dict))
        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        model_dict.update(pretrained_dict)
        super(WSDAN_CAL, self).load_state_dict(model_dict)

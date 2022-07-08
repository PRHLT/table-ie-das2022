from __future__ import print_function
from __future__ import division
import torch_geometric.transforms as T
import random
# import os
import math

import numpy as np
import torch

# import logging


def build_transforms(opts, train=True):
    tr = []
    min_elements = 0.3
    if train:
        tr.append(PosTransform(max_movement=0.2, p=opts.trans_prob, axis=0, min_elements=min_elements)) # x
        tr.append(PosTransform(max_movement=0.2, p=opts.trans_prob, axis=1, min_elements=min_elements)) #y
        tr.append(SizeTransform(max_movement=0.6, p=opts.trans_prob, axis=2, min_elements=min_elements)) #w
        tr.append(SizeTransform(max_movement=0.6, p=opts.trans_prob, axis=3, min_elements=min_elements)) #h
        tr.append(SizeTransformEdge(max_movement=0.15, p=opts.trans_prob, axis=0, min_elements=min_elements)) # edge length
        tr.append(SizeTransformEdge(max_movement=0.2, p=opts.trans_prob, axis=1, min_elements=min_elements)) # edge punct
        tr.append(PosTransformEdge(max_movement=0.1, p=opts.trans_prob, axis=2, min_elements=min_elements)) # edge x_coord_o
        tr.append(PosTransformEdge(max_movement=0.1, p=opts.trans_prob, axis=3, min_elements=min_elements)) # edge y_coord_o
        tr.append(PosTransformEdge(max_movement=0.1, p=opts.trans_prob, axis=4, min_elements=min_elements)) # edge x_coord_t
        tr.append(PosTransformEdge(max_movement=0.1, p=opts.trans_prob, axis=5, min_elements=min_elements)) # edge y_coord_t
        tr.append(PosTransformEdge(max_movement=0.1, p=opts.trans_prob, axis=6, min_elements=min_elements)) # edge x_coord_mid
        tr.append(PosTransformEdge(max_movement=0.1, p=opts.trans_prob, axis=7, min_elements=min_elements)) # edge y_coord_mid
    t = T.Compose(tr)
    return t

class PosTransform(T.BaseTransform):

    def __init__(self, max_movement=0.005, p=0.5, axis=0, min_elements=0.3) -> None:
        super().__init__()
        self.max_movement = max_movement
        self.axis = axis
        self.p = p
        self.min_elements = min_elements

    def __call__(self, data):
        if random.random() < self.p:
            if len(data.x.shape) < 2:
                return data
            a = torch.Tensor(data.x.shape[0])
            a.uniform_(-self.max_movement,self.max_movement)
            mask = torch.Tensor(data.x.shape[0])
            mask.uniform_(0,1)
            mask = mask <= self.min_elements
            a = a * mask
            x = data.x 
            x[...,self.axis] = torch.clamp(x[...,self.axis] + a, 0, 1)
            data.x = x
        return data

class PosTransformEdge(T.BaseTransform):

    def __init__(self, max_movement=0.005, p=0.5, axis=0, min_elements=0.3) -> None:
        super().__init__()
        self.max_movement = max_movement
        self.axis = axis
        self.p = p
        self.min_elements = min_elements

    def __call__(self, data):
        if random.random() < self.p:
            if len(data.x.shape) < 2:
                return data
            a = torch.Tensor(data.edge_attr.shape[0])
            a.uniform_(-self.max_movement,self.max_movement)
            mask = torch.Tensor(data.edge_attr.shape[0])
            mask.uniform_(0,1)
            mask = mask <= self.min_elements
            a = a * mask
            x = data.edge_attr 
            x[...,self.axis] = torch.clamp(x[...,self.axis] + a, 0, 1)
            data.edge_attr = x # Not needed
        return data

class SizeTransformEdge(T.BaseTransform):

    def __init__(self, max_movement=0.005, p=0.5, axis=0, min_elements=0.3) -> None:
        super().__init__()
        self.max_movement = max_movement
        self.axis = axis
        self.p = p
        self.min_elements = min_elements

    def __call__(self, data):
        if random.random() < self.p:
            if len(data.x.shape) < 2:
                return data
            a = torch.Tensor(data.edge_attr.shape[0])
            a.uniform_(-self.max_movement,self.max_movement)
            mask = torch.Tensor(data.edge_attr.shape[0])
            mask.uniform_(0,1)
            mask = mask <= self.min_elements
            a[mask] = 1
            x = data.edge_attr 
            x[...,self.axis] = torch.clamp(x[...,self.axis] * a, 0, 1)
            data.edge_attr = x # Not needed
        return data

class SizeTransform(T.BaseTransform):

    def __init__(self, max_movement=0.005, p=0.5, axis=0, min_elements=0.3) -> None:
        super().__init__()
        self.max_movement = max_movement
        self.axis = axis
        self.p = p
        self.min_elements = min_elements

    def __call__(self, data):
        if random.random() < self.p:
            if len(data.x.shape) < 2:
                return data
            a = torch.Tensor(data.x.shape[0])
            a.uniform_(-self.max_movement,self.max_movement)
            mask = torch.Tensor(data.x.shape[0])
            mask.uniform_(0,1)
            mask = mask <= self.min_elements
            a[mask] = 1
            x = data.x 
            x[...,self.axis] = torch.clamp(x[...,self.axis] * a, 0, 1)
            data.x = x
        return data
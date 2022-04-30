import torch
import torch.nn as nn
import torch.tensor as tensor
import threading
import time
import einops

import sys
import sparse_conv

__all__ = ['SparseOp']

my_conv = sparse_conv.MyConv

def split_weights(net):
    decay = []
    no_decay = []
    for m in net.modules():
        if isinstance(m, my_conv):            
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

class SparseOp():
    def __init__(self, model):
        self.model = model
        self.mask = []
        count_targets = -1
        self.bin_range = []
        for m in model.named_modules():
            if isinstance(m[1], my_conv) or isinstance(m[1], nn.Conv2d):
                count_targets += 1
                if count_targets != 0:
                    self.bin_range.append(count_targets)

        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []

        index = -1
        for m in model.named_modules():          
            if isinstance(m[1], my_conv) or isinstance(m[1], nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m[1].weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m[1].weight)

    def feedforward_sparse(self):
        self.clear_mask_list()
        self.save_params()         
        ind = -1
        for m in self.model.named_modules():
            if isinstance(m[1], my_conv) or isinstance(m[1], nn.Conv2d):
                ind = ind + 1
                if ind in self.bin_range:
                    shape = m[1].weight.data.shape                
                    w_2d = m[1].weight.data.view(-1, 4)
                    index = torch.argsort(w_2d.abs(), dim=1)[:, :2]

                    mask_ = torch.ones(w_2d.shape, device=w_2d.device)
                    mask_ = mask_.scatter_(dim=1, index=index, value=0)

                    assert w_2d.shape == mask_.shape
                    w_2d = torch.mul(w_2d, mask_)
                    self.mask.append(mask_.view(shape))
                    m[1].weight.data = w_2d.view(shape)
    
    def backward_sparse(self): 
        ind = -1                   
        for m in self.model.named_modules():
            if isinstance(m[1], my_conv) or isinstance(m[1], nn.Conv2d):
                ind = ind + 1
                if ind in self.bin_range:                
                    shape = m[1].weight.data.shape
                    w_2d = m[1].weight.data.view(-1, 4)
                    
                    assert (w_2d.shape[0] % 4) == 0
                    w_2d_t = einops.rearrange(w_2d, '(a w)(b h) ->(a b) (w h)', w=4,h=1)

                    index = torch.argsort(w_2d_t.abs(), dim=1)[:, :2]

                    mask_ = torch.ones(w_2d_t.shape, device=w_2d_t.device)
                    mask_ = mask_.scatter_(dim=1, index=index, value=0)

                    assert w_2d_t.shape == mask_.shape
                    w_2d_t = torch.mul(w_2d_t, mask_)

                    w_2d = einops.rearrange(w_2d_t, '(a b) (w h)->(a w)(b h)', w=4, b=4)
                    m[1].weight.data = w_2d.view(shape)

    def clear_mask_list(self):
        self.mask.clear()

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateSparseGradWeight(self):
        for index in range(self.num_of_params):        
            m = self.target_modules[index].grad.data
            self.target_modules[index].grad.data = m + 0.0002*(1 - self.mask[index]) * self.saved_params[index]

class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
    
    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        elif self.reduction == 'mean':
            return torch.mean(loss)
        
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')
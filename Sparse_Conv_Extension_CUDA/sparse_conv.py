import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
import SparseConv_cuda as sparse_conv_ext

class MyConvFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                im2col_step=256):
        if input is not None and input.dim() != 4:
            raise ValueError(
                'Expected 4D tensor as input, got {}D tensor instead.'.format(
                    input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, weight)

        output = input.new_empty(
            MyConvFunction._output_size(input, weight, ctx.padding,
                                            ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            sparse_conv_ext.sparse_conv_forward(
                input, weight, output, ctx.bufs_[0], ctx.bufs_[1],
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.groups,
                cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
                sparse_conv_ext.sparse_conv_backward_input(
                    input, grad_output, grad_input,
                    weight, ctx.bufs_[0], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.groups,
                    cur_im2col_step)

            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
                sparse_conv_ext.sparse_conv_backward_parameters(
                    input, grad_output,
                    grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.groups, 1,
                    cur_im2col_step, 0.03)

        return (grad_input, grad_weight, None, None, None, None, None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be {})'.format(
                    'x'.join(map(str, output_size))))
        return output_size

sparse_conv = MyConvFunction.apply

class MyConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(MyConv, self).__init__()

        #assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))
        if bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        input_pad = (
            x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        out = sparse_conv(x, self.weight, self.stride, self.padding,
                          self.dilation, self.groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()

        if self.bias is not None:
            b,c,h,w = out.shape
            out = out + self.bias.view(1, c, 1, 1)
        return out

if __name__ == '__main__':
    sconv = MyConv(1,2,3).cuda(torch.device("cuda:3"))
    conv = torch.nn.Conv2d(1,2,3,bias=False).cuda(torch.device("cuda:3"))
    sconv.weight.data = conv.weight.data

    '''
    check forward
    '''
    x=torch.randn(64,10,224,224).cuda(torch.device("cuda:4"))
    y=sconv(x)
    z=conv(x)
    print((z-y).norm().sum())
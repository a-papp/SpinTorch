"""
Binarization layer with pseudo-gradient

The forward function returns the sign of the tensor.
The backward function returns the gradients unaltered.
"""
import torch

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return x.sign()
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

def binarize(x):
    return Binarize.apply(x)
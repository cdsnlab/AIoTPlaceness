import torch
import torch.nn as nn

class PTanh(nn.Module):
    '''
    Applies the Penalized Tanh (PTanh) function element-wise:
        ptanh(x) = tanh(x) if x > 0 or p * tanh(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = PTanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self, penalty=0.25):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.penalty = penalty # init penalty

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return torch.where(input > 0, torch.tanh(input), self.penalty * torch.tanh(input))
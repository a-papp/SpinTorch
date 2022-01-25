"""Class to calculate the exchange field"""

from torch import nn, tensor


class Exchange(nn.Module):
    A_exch = 3.65e-12       # exchange coefficient (J/m)

    def __init__(self, d: tuple):
        super().__init__()
        
        # defining the LAPLACE convolution kernel for exchange field
        self.LAPLACE = nn.Conv2d(3, 3, 3, groups=3, padding=1, padding_mode='replicate', bias=False)
        self.LAPLACE.weight.requires_grad = False
        idx2, idy2 = 1.0/d[0]**2, 1.0/d[1]**2
        self.LAPLACE.weight[:,] = tensor([[[0.0,         idx2,    0.0 ],
                                           [idy2, -2*(idx2+idy2), idy2],
                                           [0.0,         idx2,    0.0 ]]])

    def forward(self, m, Msat):
        """
        Calculate the exchange field of magnetization m.
        
        Inputs: m normalized magnetization (pytorch tensor)
                Msat saturation magnetization (pytorch tensor)
        Outputs: exchange field (same size as m)
        """
        
        B_exch = 2*self.A_exch/Msat * self.LAPLACE(m)
        B_exch[:,:,Msat == 0] = 0   # handle division by 0
        return B_exch



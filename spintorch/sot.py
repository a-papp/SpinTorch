"""
Class to calculate the spin-orbit torque

based on:
Wang, Z., Sun, Y., Wu, M., Tiberkevich, V., & Slavin, A. (2011). 
Control of spin waves in a thin film ferromagnetic insulator through interfacial spin scattering. 
Physical review letters, 107(14), 146602.
"""

from torch import nn, cross, tensor
from torch.nn.functional import normalize

class SOT(nn.Module):
    gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
    hbar = 6.62607015e-34   # Planck's constant
    qe   = 1.60217662e-19   # electron charge
    C_SOT = 1e-6    # phenomenological SOT coefficient ()
    T_SOT = 20e-9   # effective SOT thickness (m)
    J_SOT = -4e8    # spin current density (A/m^2)
    sigma = [0.0,1.0,0.0] # spin polarization vector
    active = True
    
    def __init__(self, dim: tuple):
        super().__init__()

        self.dim = dim
                 
        sigma_n = normalize(tensor(self.sigma).view(1,-1,1,1))
        sigma_SOT = sigma_n.expand((-1,-1) + dim).clone()
        self.register_buffer("sigma_SOT", sigma_SOT)
        

    def forward(self, m, Msat):
        """Calculate spin-orbit torque"""
        if self.active:
            t_SOT = (self.gamma_LL*self.C_SOT*self.J_SOT*self.hbar/(2*self.qe)
                     /self.T_SOT/Msat**2 * cross(m, cross(self.sigma_SOT, m,1),1))
            t_SOT[:,:,Msat == 0] = 0
            return t_SOT
        else:
            return 0




"""Modules for representing the trained parameters"""

import torch
from torch import nn, sum, tensor, zeros, ones, real
from torch.fft import fftn, ifftn
from .demag import Demag
from .binarize import binarize
from numpy import pi



class WaveGeometry(nn.Module):
    def __init__(self, dim: tuple, d: tuple, B0: float, Ms: float):
        super().__init__()

        self.dim = dim
        self.d   = d
        self.register_buffer("B0", tensor(B0))
        self.register_buffer("Ms", tensor(Ms))

    def forward(self):
        raise NotImplementedError


class WaveGeometryFreeForm(WaveGeometry):
    def __init__(self, dim: tuple, d: tuple, B0: float, B1: float, Ms: float):

        super().__init__(dim, d, B0, Ms)

        self.rho = nn.Parameter(zeros(dim))
        self.register_buffer("B", zeros((3,)+dim))
        self.register_buffer("B1", tensor(B1))
        self.B[1,] = self.B0
        
    def forward(self):
        self.B = torch.zeros_like(self.B)
        self.B[1,] = self.B1*self.rho + self.B0
        return self.B



class WaveGeometryMs(WaveGeometry):
    def __init__(self, dim: tuple, d: tuple, Ms: float, B0: float):

        super().__init__(dim, d, B0, Ms)

        self.rho = nn.Parameter(ones(dim))
        self.register_buffer("Msat", zeros(dim))
        self.register_buffer("B0", tensor(B0))
        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] = self.B0
        
    def forward(self):
        self.Msat = self.Ms*self.rho
        return self.Msat


class WaveGeometryArray(WaveGeometry):
    def __init__(self, rho, dim: tuple, d: tuple, Ms: float, B0: float,
                  r0: int, dr: int, dm: int, z_off: int, rx: int, ry: int,
                  Ms_CoPt: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.dm = dm
        self.z_off = z_off
        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_CoPt", tensor(Ms_CoPt))
        self.rho = nn.Parameter(rho.clone().detach())
        self.convolver = nn.Conv2d(3, 3, self.dm, padding=(self.dm//2),
                                    groups=3, bias=False)
        self.convolver.weight.requires_grad = False
        
        for i in range(3):
            self.convolver.weight[i, 0, ] = ones((dm, dm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry = self.r0, self.dr, self.rx, self.ry
        rho_binary = binarize(self.rho)   
        m_rho = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho_binary
        m_rho_ = self.convolver(m_rho)[:,:,0:nx,0:ny]
        m_ = nn.functional.pad(m_rho_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        
        self.B = B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]*self.Ms_CoPt*mu0
        self.B[1,] += self.B0
        return self.B

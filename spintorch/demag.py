"""
Class to calculate the demagnetization kernel and field

based on:
Ru Zhu, Accelerate micromagnetic simulations with GPU programming in MATLAB
https://arxiv.org/ftp/arxiv/papers/1501/1501.07293.pdf
"""

from numpy import pi, log, arctan, sqrt
import numpy as np
from numba import jit
import torch
from torch import nn, sum, tensor, real
from torch.fft import fft2, ifft2, fftn


class Demag(nn.Module):
    def __init__(self, dim: tuple, d: tuple):
        super().__init__()

        self.dim = dim
        self.d   = d

        
        Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft = self.demag_tensor_fft_2D()
        self.register_buffer("Kxx_fft", Kxx_fft)
        self.register_buffer("Kyy_fft", Kyy_fft)
        self.register_buffer("Kzz_fft", Kzz_fft)
        self.register_buffer("Kxy_fft", Kxy_fft)

    def forward(self, m, Msat):
        """
        Calculate the demag field of magnetization m.
        
        Inputs: m normalized magnetization (pytorch tensor)
                Msat saturation magnetization (pytorch tensor)
        Outputs: demagnetization field (same size as m)
        """

        M_ = nn.functional.pad(m*Msat, (0, self.dim[1], 0, self.dim[0]))
        M_fft = (fft2(M_)) 
      
        B_demag = real(ifft2((torch.stack(
            [sum((torch.stack([self.Kxx_fft, self.Kxy_fft],1)*M_fft[:,0:2,]),1),
             sum((torch.stack([self.Kxy_fft, self.Kyy_fft],1)*M_fft[:,0:2,]),1),
             (self.Kzz_fft*M_fft[:,2,])], 1))))

        return pi*4e-7*B_demag[...,self.dim[0]-1:2*self.dim[0]-1,self.dim[1]-1:2*self.dim[1]-1]


    @staticmethod
    @jit(nopython=True)
    def demag_tensor(nx, ny, nz, dx, dy, dz, z_off=0):
        """
        Calculate the demagnetization tensor.
        Numba is used to accelerate the calculation.
        Inputs: nx, ny, nz: number of cells in x/y/z,
                dx, dy, dz: cellsizes in x/y/z,
                z_off: optional offset in z direction (integer with units of dz)
        Outputs: demag tensor elements (numpy.array)
        """
        # Initialization of demagnetization tensor
        Kxx = np.zeros((nx*2, ny*2, nz*2))
        Kyy = np.zeros((nx*2, ny*2, nz*2))
        Kzz = np.zeros((nx*2, ny*2, nz*2))
        Kxy = np.zeros((nx*2, ny*2, nz*2))
        Kxz = np.zeros((nx*2, ny*2, nz*2))
        Kyz = np.zeros((nx*2, ny*2, nz*2))
    
        for K in range(-nz+1+z_off, nz+z_off):
            for J in range(-ny+1, ny):
                for I in range(-nx+1, nx):
                    L, M, N = (I+nx-1), (J+ny-1), (K+nz-1-z_off)  # non-negative indices
                    for i in (-0.5, 0.5):
                        for j in (-0.5, 0.5):
                            for k in (-0.5, 0.5):
                                sgn = (-1)**(i+j+k+1.5)/(4*pi)
                                r = sqrt(((I+i)*dx)**2 + ((J+j)*dy)**2 + ((K+k)*dz)**2)
                                Kxx[L, M, N] += sgn * arctan((K+k)*(J+j)*dz*dy/(r*(I+i)*dx))
                                Kyy[L, M, N] += sgn * arctan((I+i)*(K+k)*dx*dz/(r*(J+j)*dy))
                                Kzz[L, M, N] += sgn * arctan((J+j)*(I+i)*dy*dx/(r*(K+k)*dz))
                                Kxy[L, M, N] -= sgn * log(abs((K+k)*dz + r))
                                Kxz[L, M, N] -= sgn * log(abs((J+j)*dy + r))
                                Kyz[L, M, N] -= sgn * log(abs((I+i)*dx + r))
    
        return Kxx, Kyy, Kzz, Kxy, Kxz, Kyz
    
    
    @staticmethod
    @jit(nopython=True)
    def demag_tensor_2D(nx, ny, dx, dy, dz):
        """
        Calculate the demagnetization tensor for 2D problems.
        
        Numba is used to accelerate the calculation.
        Inputs: nx, ny: number of cells in x/y,
                dx, dy, dz: cellsizes in x/y/z,
        Outputs: demag tensor elements (numpy.array)
        """
        # Initialization of demagnetization tensor
        Kxx = np.zeros((nx*2, ny*2))
        Kyy = np.zeros((nx*2, ny*2))
        Kzz = np.zeros((nx*2, ny*2))
        Kxy = np.zeros((nx*2, ny*2))
        K = 0
        for J in range(-ny+1, ny):
            for I in range(-nx+1, nx):
                L, M = (I+nx-1), (J+ny-1)  # non-negative indices
                for i in (-0.5, 0.5):
                    for j in (-0.5, 0.5):
                        for k in (-0.5, 0.5):
                            sgn = (-1)**(i+j+k+1.5)/(4*pi)
                            r = sqrt(((I+i)*dx)**2 + ((J+j)*dy)**2 + ((K+k)*dz)**2)
                            Kxx[L, M] += sgn * arctan((K+k)*(J+j)*dz*dy/(r*(I+i)*dx))
                            Kyy[L, M] += sgn * arctan((I+i)*(K+k)*dx*dz/(r*(J+j)*dy))
                            Kzz[L, M] += sgn * arctan((J+j)*(I+i)*dy*dx/(r*(K+k)*dz))
                            Kxy[L, M] -= sgn * log(abs((K+k)*dz + r))
    
        return Kxx, Kyy, Kzz, Kxy
    
    
    def demag_tensor_fft(self, z_off=0):
        """
        Return the demagnetization kernel in Fourier domain.
        
        Inputs: z_off: optional offset in z direction (integer with units of dz)
        Outputs: demag tensor elements stacked (torch.tensor)
        """
        Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = self.demag_tensor(
            self.dim[0], self.dim[1], 1, self.d[0], self.d[1], self.d[2], z_off)
 
        Kx_fft = fftn(tensor(np.stack((Kxx,Kxy,Kxz),0),dtype=torch.float32).unsqueeze(0), dim=(2,3))
        Ky_fft = fftn(tensor(np.stack((Kxy,Kyy,Kyz),0),dtype=torch.float32).unsqueeze(0), dim=(2,3))  
        Kz_fft = fftn(tensor(np.stack((Kxz,Kyz,Kzz),0),dtype=torch.float32).unsqueeze(0), dim=(2,3))   

        return Kx_fft, Ky_fft, Kz_fft
    
    
    def demag_tensor_fft_2D(self):
        """
        Return the demagnetization kernel in Fourier domain.
        
        Symmetries in 2D: Kyx=Kxy, Kxz=Kzx=Kyz=Kzy=0
        Inputs: self
        Outputs: demag tensor elements (exploiting symmetry) (torch.tensor)
        """
        Kxx, Kyy, Kzz, Kxy = self.demag_tensor_2D(self.dim[0], self.dim[1], self.d[0], self.d[1], self.d[2])

        Kxx_fft = fft2(tensor(Kxx, dtype=torch.float32).unsqueeze(0))
        Kyy_fft = fft2(tensor(Kyy, dtype=torch.float32).unsqueeze(0))
        Kzz_fft = fft2(tensor(Kzz, dtype=torch.float32).unsqueeze(0))
        Kxy_fft = fft2(tensor(Kxy, dtype=torch.float32).unsqueeze(0))
        
        return Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft
    

"""Micromagnetic solver with backpropagation"""

import torch 
from torch import nn, cat, cross, tensor, zeros, empty
from torch.utils.checkpoint import checkpoint
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
from math import ceil
from .geom import WaveGeometryMs 
from .demag import Demag 
from .exch import Exchange
from .damping import Damping
from .sot import SOT 

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class MMSolver(nn.Module):
    gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
    relax_timesteps = 100
    retain_history = False
    def __init__(self, geometry, dt: float, sources=[], probes=[]):
        super().__init__()

        self.register_buffer("dt", tensor(dt))  # timestep (s)
        self.input = None  # input signal

        self.geom = geometry
        self.sources = nn.ModuleList(sources)
        self.probes = nn.ModuleList(probes)
        self.demag_2D = Demag(self.geom.dim, self.geom.d)
        self.exch_2D = Exchange(self.geom.d)
        self.Alpha = Damping(self.geom.dim)
        self.torque_SOT = SOT(self.geom.dim)
        SOT.gamma_LL = self.gamma_LL

        self.register_buffer("B_ext0", self.geom.B)  # input signal
        self.register_buffer("Msat", self.geom() )  # initial Ms

        m0 = zeros((1, 3,) + self.geom.dim)
        m0[:, 1,] =  1     # set initial magnetization in y direction
        self.m_history = []
        self.register_buffer("m0", m0)
        self.fwd = False  # differentiates main fwd pass and checpointing runs

    def forward(self, input):
        self.m_history = []
        self.fwd = True
        if isinstance(self.geom, WaveGeometryMs):
            self.Msat = self.geom()
            self.B_ext0 = self.geom.B
        else:
            self.B_ext0 = self.geom()
            self.Msat = self.geom.Ms
        
        self.input = None
        self.relax() # relax magnetization and store in m0 (no gradients)
        self.input = input
        # outputs = self.run(self.m0) # run the simulation

        t_save = torch.arange(0, 601, device=self.dt.device)*self.dt
        outputs = self.odeint_adjoint_step(self.m0, t_save)
        # print(len(outputs))
        # print(outputs[0].size())
        # exit()
        self.fwd = False
        return outputs

    def run(self, m):
        """Run the simulation in multiple stages for checkpointing"""
        outputs = []
        timesteps = 600
        N = ceil(np.sqrt(timesteps)) # number of stages 
        for stg in range(0, N):
            n_start = stg*N
            if n_start >= timesteps:
                break
            n_end = min((stg+1)*N,timesteps)
            output, m = checkpoint(self.run_stage, m, n_start, n_end, use_reentrant=False)
            outputs.append(output)
        return cat(outputs, dim=1)
        
    def run_stage(self, m, n_start,n_end):
        """Run a subset of timesteps (needed for 2nd level checkpointing)"""
        outputs = empty(0,device=self.dt.device)
        # Loop through the signal 
        for n in range(n_start,n_end):
            # Propagate the fields (with checkpointing to save memory)
            m = checkpoint(self.rk4_step, n*self.dt, m, use_reentrant=False)
            # Measure the outputs (checkpointing helps here as well)
            outputs = checkpoint(self.measure_probes, m, outputs, use_reentrant=False)
            if self.retain_history and self.fwd:
                self.m_history.append(m.detach().cpu())
        return outputs, m
        
    def inject_sources(self, sig): 
        """Add the excitation signal components to B_ext"""
        for i, src in enumerate(self.sources):
            B_ext = src(self.B_ext0, sig)
        return B_ext

    def measure_probes(self, m, outputs): 
        """Extract outputs and concatenate to previous values"""
        probe_values = []
        for probe in self.probes:
            probe_values.append(probe((m-self.m0)*self.Msat))
        outputs = cat([outputs, cat(probe_values).unsqueeze(0).unsqueeze(0)], dim=1)
        return outputs
        
    def relax(self): 
        """Run the solver with high damping to relax magnetization"""
        with torch.no_grad():
            for n in range(self.relax_timesteps):
                self.m0 = self.rk4_step(0, self.m0, relax=True)

    def odeint_adjoint_step(self, m, t_save, relax=False):
        # Define update function     
        def update_fn(tg,m):
            return self.step_fn(tg/self.gamma_LL,m,relax)

        # Call odeint_adjoint with dopri5
        out = odeint_adjoint(update_fn,m,t_save*self.gamma_LL,adjoint_params=self.parameters(),method="dopri5",atol=1e-6)
        outputs = empty(0,device=self.dt.device)
        for i, m in enumerate(out):
            # checkpointing is important here
            outputs = checkpoint(self.measure_probes, m, outputs, use_reentrant=False)
            if self.retain_history and self.fwd:
                self.m_history.append(m.detach().cpu())
        return outputs

    
    def rk4_step(self, t, m, relax=False):
        """Implement a 4th-order Runge-Kutta solver"""
        dt = self.dt
        h  = self.gamma_LL * dt    # this time unit is closer to 1
        k1 = self.step_fn(t,        m,          relax)
        k2 = self.step_fn(t + dt/2, m + h*k1/2, relax)
        k3 = self.step_fn(t + dt/2, m + h*k2/2, relax)
        k4 = self.step_fn(t + dt,   m + h*k3,   relax)
        return (m + h/6 * ((k1 + 2*k2) + (2*k3 + k4)))
    
    def step_fn(self,t,m,relax):
        Bext = self.inject_sources(self.signal(t))
        return self.torque_LLG(m, Bext, self.Msat, relax)  
          
    def signal(self,t):
        if self.input is None:
            sig = 0
        elif callable(self.input):
            sig = self.input(t)
        else:
            print("Non-callable signal not implemented!")
        return sig
                    
    def B_eff(self, m, B_ext, Msat): 
        """Sum the field components to return the effective field"""
        return B_ext + self.exch_2D(m,Msat) + self.demag_2D(m,Msat) 
        
    def torque_LLG(self, m, B_ext, Msat, relax=False):
        """Calculate Landau-Lifshitz-Gilbert torque (= dm/dt)"""
        m_x_Beff = cross(m, self.B_eff(m, B_ext, Msat),1)
        return (-(1/(1+self.Alpha(relax)**2) * (m_x_Beff + self.Alpha(relax)*cross(m, m_x_Beff,1))) 
                + self.torque_SOT(m, Msat))

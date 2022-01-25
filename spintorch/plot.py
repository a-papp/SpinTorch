import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
from .geom import WaveGeometryMs, WaveGeometry
from .solver import MMSolver

import warnings
warnings.filterwarnings("ignore", message=".*No contour levels were found.*")


mpl.use('Agg',) # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600


def plot_loss(loss_iter, plotdir):
    fig = plt.figure()
    plt.plot(loss_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'loss.png')
    plt.close(fig)
    
def plot_output(u, p, epoch, plotdir):
    fig = plt.figure()
    plt.bar(range(1,1+u.size()[0]), u.detach().cpu().squeeze(), color='k')
    plt.xlabel("output number")
    plt.ylabel("output")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'output_epoch%d_X%d.png' % (epoch, p))
    plt.close(fig)


def _plot_probes(probes, ax):
    markers = []
    for i, probe in enumerate(probes):
        x,y = probe.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='k',markersize=4,alpha=0.8)
        markers.append(marker)
    return markers


def _plot_sources(sources, ax):
    markers = []
    for i, source in enumerate(sources):
        x,y = source.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='g',markersize=4,alpha=0.8)
        markers.append(marker)
    return markers


def geometry(model, ax=None, outline=False, outline_pml=True, epoch=0, plotdir=''):

    geom = model.geom
    probes = model.probes
    sources = model.sources
    A = model.Alpha()[0, 0, ].squeeze()
    alph = A.min().cpu().numpy()
    B = geom.B[1,].detach().cpu().numpy().transpose()

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    markers = []
    if not outline:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            h1 = ax.imshow(Msat, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label='Saturation magnetization (A/m)')
        else:
            h1 = ax.imshow(B*1e3, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label='Magnetic field (mT)')
    else:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            ax.contour(Msat, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)
        else:
            ax.contour(B, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        ax.contour(b_boundary, levels=[alph*1.0001], colors=['k'], linestyles=['dotted'], linewidths=[0.75], alpha=1)

    markers += _plot_probes(probes, ax)
    markers += _plot_sources(sources, ax)
        
    if plotdir:
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch))
        plt.close(fig)


def wave_integrated(model, m_history, filename=''):
    
    m_int = m_history.pow(2).sum(dim=0).numpy().transpose()
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    vmax = m_int.max()
    h = ax.imshow(m_int, cmap=plt.cm.viridis, origin="lower", norm=LogNorm(vmin=vmax*0.01,vmax=vmax))
    plt.colorbar(h)
    geometry(model, ax=ax, outline=True)

    if filename:
        fig.savefig(filename)
        plt.close(fig)


def wave_snapshot(model, m_snap, filename='', clabel='m'):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    m_t = m_snap.cpu().numpy().transpose()
    h = axs.imshow(m_t, cmap=plt.cm.RdBu_r, origin="lower", norm=CenteredNorm())
    geometry(model, ax=axs, outline=True)
    plt.colorbar(h, ax=axs, label=clabel, shrink=0.80)
    axs.axis('image')
    if filename:
        fig.savefig(filename)
        plt.close(fig)
        

#!/usr/bin/env python3
"""Illustrative 2D off-state electrostatic model of a trench-gate 4H-SiC JFET.

The Green operator is the inverse of a sparse finite-volume/-difference
negative-Laplacian with mixed boundary conditions. This is NOT a semiconductor
TCAD avalanche solution. The device drawing provides topology, not dimensions;
therefore all numerical geometry, doping and bias values are ASSUMPTIONS.

Requires: numpy scipy matplotlib
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
q = 1.602176634e-19
EPS_0 = 8.8541878128e-12
EPS_SIC = 9.7 * EPS_0                 # F/m, isotropic approximation
ND = 5.0e15                          # cm^-3, assumed n- drift
E_CRIT = 2.5                          # MV/cm, illustrative critical field
VG = -15.0                           # V, source-referenced gate bias
VS = 0.0
LZ = 13.0                            # um (12.4 um effective n- drift)
LX = 5.0                             # half-width um, mirror boundaries
NX, NZ = 201, 261                   # grid; dx=dz=0.05 um
TRENCH_CENTERS = [-2.55, 2.55]       # um
TRENCH_HALF_WIDTH = 0.68            # um; metal width 1.36 um
TRENCH_ZTOP = 0.55                   # um
TRENCH_ZBOTTOM = 2.45                # um
PPLUS_WIDTHS = [0.20, 0.35, 0.50, 0.65, 0.80] # um
BASE_W = 0.50                        # um
PPLUS_BOTTOM_THICKNESS = 0.45        # um; held fixed in width sweep
SMOOTH_UM = 0.20                     # field averaging length; a MODEL assumption
COMPARE_VD = 800.0                   # V, common comparison drain bias

x = np.linspace(-LX, LX, NX)
z = np.linspace(0, LZ, NZ)
X, Z = np.meshgrid(x, z)
DX = (x[1]-x[0]) * 1e-6
DZ = (z[1]-z[0]) * 1e-6
W_X = 1 / DX**2
W_Z = 1 / DZ**2


def geometry(width):
    """Gate trench, P+ collars, source and drain Dirichlet geometry."""
    gate = np.zeros_like(X, dtype=bool)
    pplus = np.zeros_like(X, dtype=bool)
    for cx in TRENCH_CENTERS:
        trench = (np.abs(X-cx) <= TRENCH_HALF_WIDTH) & (Z >= TRENCH_ZTOP) & (Z <= TRENCH_ZBOTTOM)
        gate |= trench
        # Only lateral P+ width is swept; bottom thickness is held FIXED.
        # Elliptically rounded P+ collar near bottom trench corners.
        dxr = np.maximum(np.abs(X-cx) - TRENCH_HALF_WIDTH, 0.0)
        zbelow = np.maximum(Z - TRENCH_ZBOTTOM, 0.0)
        bottom = ((dxr/width)**2 + (zbelow/PPLUS_BOTTOM_THICKNESS)**2 <= 1.0) & (Z >= TRENCH_ZBOTTOM)
        # Gate sidewall P+ width tapers near the top source plane.
        side_scale = np.clip((Z - 0.64)/0.35, 0.0, 1.0)
        side = ((np.abs(X-cx) > TRENCH_HALF_WIDTH) &
                (np.abs(X-cx) <= TRENCH_HALF_WIDTH + width*side_scale) &
                (Z >= 0.65) & (Z < TRENCH_ZBOTTOM))
        pplus |= (side | bottom) & (~trench)
    # Top outside metal-filled trenches -> contacted n+ source
    source = (Z == z[0]) & (~((np.abs(X-TRENCH_CENTERS[0]) <= TRENCH_HALF_WIDTH+0.10) |
                              (np.abs(X-TRENCH_CENTERS[1]) <= TRENCH_HALF_WIDTH+0.10)))
    drain = (Z == z[-1])
    fixed = source | drain | gate | pplus
    return dict(source=source, drain=drain, gate=gate, pplus=pplus, fixed=fixed)


def sparse_green_system(width):
    """A inverse implements the discrete Green's operator on free nodes."""
    m = geometry(width)
    fixed = m['fixed']
    free = ~fixed
    index = -np.ones(X.shape, dtype=np.int32)
    index[free] = np.arange(np.count_nonzero(free), dtype=np.int32)
    nfree = np.count_nonzero(free)
    rows, cols, vals = [], [], []
    rhs_base = np.zeros(nfree, dtype=np.float64)
    rhs_unit_drain = np.zeros(nfree, dtype=np.float64)

    # Free semiconductor charge: uniform fully-ionized donors except near n+ source
    charge_rhs = (q * ND * 1e6 / EPS_SIC) * (Z >= 0.60)  # V/m^2
    rhs_base[:] = charge_rhs[free]
    bias = np.zeros_like(X)
    bias[m['gate'] | m['pplus']] = VG
    bias[m['source']] = VS
    # bottom drain potential added separately as 1-V basis
    z_nodes, x_nodes = np.where(free)
    offsets = [(-1, 0, W_Z), (1, 0, W_Z), (0, -1, W_X), (0, 1, W_X)]
    for k, (iz, ix) in enumerate(zip(z_nodes, x_nodes)):
        diag = 0.0
        for dzstep, dxstep, weight in offsets:
            zz, xx = iz+dzstep, ix+dxstep
            if not (0 <= zz < NZ and 0 <= xx < NX):
                # no neighbor: zero normal E = homogeneous Neumann BC
                continue
            diag += weight
            if fixed[zz, xx]:
                if m['drain'][zz, xx]:
                    rhs_unit_drain[k] += weight
                else:
                    rhs_base[k] += weight*bias[zz, xx]
            else:
                rows.append(k); cols.append(int(index[zz, xx])); vals.append(-weight)
        rows.append(k); cols.append(k); vals.append(diag)
    A = coo_matrix((vals,(rows,cols)), shape=(nfree,nfree)).tocsc()
    lu = splu(A)
    # solve -laplace V0 = charge plus fixed source/gate values, with VD=0
    vb = np.zeros_like(X)
    vb[m['gate'] | m['pplus']] = VG
    vb[free] = lu.solve(rhs_base)
    vd = np.zeros_like(X)
    vd[m['drain']] = 1.0
    vd[free] = lu.solve(rhs_unit_drain)
    # residual check in the free region, scaled against rhs norm
    residual = np.max(np.abs(A@vb[free]-rhs_base)) / max(np.max(np.abs(rhs_base)),1)
    return m, vb, vd, residual


def field(V, m):
    """Electric field in MV/cm; exclude equipotential contact interiors."""
    dv_dz, dv_dx = np.gradient(V, DZ, DX, edge_order=2)
    ex = -dv_dx * 1e-8
    ez = -dv_dz * 1e-8
    em = np.hypot(ex, ez)
    active = (~m['fixed']) & (Z >= 0.75) & (Z <= LZ-0.2) & (np.abs(X)< LX-0.1)
    return ex, ez, em, active


def averaged_E(E, active):
    """Regularize interface staircasing by averaging |E| over 0.20 um.

    This is a numerical field-criterion proxy, not an avalanche ionization law.
    The physical averaging length is held fixed during mesh refinement.
    """
    sigmas = (SMOOTH_UM/(DZ*1e6), SMOOTH_UM/(DX*1e6))
    weights = gaussian_filter(active.astype(float), sigma=sigmas)
    weighted = gaussian_filter(np.where(active, E, 0.0),sigma=sigmas)
    smooth = weighted / np.maximum(weights, 1e-12)
    good = active & (weights > 0.50)
    return smooth, good


def get_critical_voltage(m, v0, v1):
    ex0, ez0, _, active = field(v0, m)
    ex1, ez1, _, _ = field(v1, m)
    def peak(Vd):
        em = np.hypot(ex0+Vd*ex1, ez0+Vd*ez1)
        effective_E, valid = averaged_E(em, active)
        return np.max(effective_E[valid])
    if peak(0.0) >= E_CRIT:
        raise RuntimeError('Threshold exceeded even at 0 V drain')
    hi = 250.0
    while peak(hi)<E_CRIT and hi<20000.0:
        hi *= 2
    if peak(hi)<E_CRIT:
        raise RuntimeError('No crossing of Ecrit within numerical range')
    Vcrit = brentq(lambda vd: peak(vd)-E_CRIT, 0.0, hi, xtol=1e-4)
    return Vcrit, peak(Vcrit)


def label_region(ax, m, title, is_field=False):
    for arr, fill in [(m['pplus'],'#05a699'),(m['gate'],'#8992a3')]:
        ax.contourf(X, Z, arr.astype(int), levels=[0.5,1.5],colors=[fill],zorder=8)
    ax.axhspan(-0.03, 0.06, color='#455d74', zorder=10)
    ax.axhspan(12.93,13.045,color='#455d74',zorder=10)
    for cx in TRENCH_CENTERS:
        ax.text(cx,1.55,'G\n−15 V',ha='center',va='center',fontsize=8,color='white',zorder=12,fontweight='bold')
    ax.text(0,0.31,'S = 0 V',ha='center',va='center',fontsize=9,color='#123652',zorder=12,fontweight='bold')
    ax.text(0,5.7,'n− drift',ha='center',va='center',fontsize=10,color='#0c5f53' if not is_field else 'white',zorder=12)
    ax.text(0,12.985,'D = +V$_D$',ha='center',va='center',fontsize=8,color='white',zorder=12,fontweight='bold')
    ax.set_title(title,fontsize=12,fontweight='bold',pad=12)
    ax.set(xlim=(-5,5),ylim=(13.12,-0.13),xlabel='Lateral x (µm)', ylabel='Depth z (µm)')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=9)


def make_figures(solutions):
    base = next(s for s in solutions if np.isclose(s['width'], BASE_W))
    m=base['m']
    plotVD = 0.90*base['Vbd']
    V=base['vb'] + plotVD*base['vd']
    ex,ez,E,active=field(V,m)
    E_eff, valid = averaged_E(E,active)
    Emasked = np.ma.array(E_eff, mask=(~valid))
    Vmasked=np.ma.array(V, mask=(m['gate'] | m['pplus']))

    cmap = LinearSegmentedColormap.from_list('green_field', ['#063d31','#007a52','#16ac55','#7fdb40','#faff71'])
    cmap.set_bad('#e8f3ee')
    fig = plt.figure(figsize=(16,10),facecolor='white')
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2,3, figure=fig, height_ratios=[2.75,1.18], width_ratios=[1,1,1],hspace=0.42,wspace=0.36,
                  left=0.075,right=0.95,top=0.90,bottom=0.085)
    axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
    # Device plot
    ax=axs[0]
    ax.set_facecolor('#eaf7e5')
    ax.axhspan(0.55,12.55,color='#dff2c9',zorder=0)
    ax.axhspan(12.55,12.9,color='#8cb5db',zorder=0)
    label_region(ax,m,'A · trench-gate JFET geometry')
    for cx in TRENCH_CENTERS:
        ax.text(cx + (0.95 if cx<0 else -0.95),2.05,'p+',va='center',ha='center',color='#ffffff',
                fontsize=10,weight='bold',zorder=12)
    ax.annotate('P+ collar width\nw = 0.50 µm', xy=(TRENCH_CENTERS[1]-TRENCH_HALF_WIDTH-BASE_W,2.0),
                xytext=(0.15,4.3), arrowprops=dict(arrowstyle='->',color='#294456',lw=1.4),
                fontsize=9,color='#294456',ha='center',zorder=15)
    ax.text(0,8.1, f'N$_D$ = {ND:.1e} cm$^{{-3}}$',ha='center',fontsize=9,zorder=12)
    # field map
    ax=axs[1]
    vlim=E_CRIT
    me=ax.pcolormesh(X,Z,Emasked,cmap=cmap,vmin=0,vmax=vlim,shading='auto')
    eflevels=np.linspace(0.3,E_CRIT-0.15,8)
    try: ax.contour(X,Z,Emasked,levels=eflevels,colors='white',linewidths=0.62,alpha=0.75)
    except ValueError: pass
    label_region(ax,m,f'B · 0.20 µm avg |E| at {plotVD:.0f} V',is_field=True)
    bar=fig.colorbar(me,ax=ax,shrink=0.68,pad=0.028)
    bar.set_label('Averaged |E| (MV/cm)',fontsize=9)
    ii,jj=np.where(active & np.isfinite(E) & (E>= np.nanmax(E[active])-1e-7))
    if len(ii):
        ax.plot(x[jj[0]],z[ii[0]],marker='x',ms=10,mew=2,color='#fff9b0',zorder=15)
    # potential map
    ax=axs[2]
    pp=ax.pcolormesh(X,Z,Vmasked,cmap='turbo',shading='auto',vmin=VG,vmax=plotVD)
    try: ax.contour(X,Z,np.ma.array(V,mask=m['fixed']),levels=np.linspace(0,plotVD,10),
                    colors='#ffffff',linewidths=0.65,alpha=0.82)
    except ValueError: pass
    label_region(ax,m,f'C · potential V at {plotVD:.0f} V',is_field=True)
    bar2=fig.colorbar(pp,ax=ax,shrink=0.68,pad=0.028)
    bar2.set_label('Potential (V)',fontsize=9)
    # sweep bottom
    ax=fig.add_subplot(gs[1,:])
    ws=np.array([s['width'] for s in solutions])
    vbs=np.array([s['Vbd'] for s in solutions])
    # Conducting aperture between the P+ collars on the device centerline
    widths_um=2*TRENCH_CENTERS[1]-2*(TRENCH_HALF_WIDTH+ws)
    ref_width=widths_um[0]
    channelR=ref_width/widths_um  # geometrical inverse-area proxy, not device R_on
    ax.plot(ws,vbs,'-o',color='#0c8056',lw=2.7,ms=7,label='Ecrit-limited voltage (V)')
    ax.axvline(BASE_W,color='#6c7b8d',ls='--',lw=1)
    for w,b in zip(ws,vbs):
        ax.text(w,b+12,f'{b:.0f}',ha='center',fontsize=10,color='#075b3e',fontweight='bold')
    ax.set(xlabel='P+ lateral collar width w (µm)',ylabel='Predicted field-threshold voltage (V)')
    ax.grid(alpha=0.20)
    ax.set_xlim(ws.min()-0.06,ws.max()+0.06)
    ax.set_ylim(max(0,vbs.min()-140), vbs.max()+170)
    ax.set_title('D · P+ width sweep: Ecrit proxy and channel constriction',fontsize=13,pad=9,fontweight='bold')
    ax2=ax.twinx()
    ax2.plot(ws,channelR,'--s',color='#c36923',lw=1.9,ms=6,label='1 / aperture (relative)')
    ax2.set_ylabel('Channel resistance proxy (relative, not R$_{DS(on)}$)',color='#a24c12')
    ax2.tick_params(axis='y',labelcolor='#a24c12')
    ax2.set_ylim(0.88,1.70)
    handles=ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(handles,[h.get_label() for h in handles],loc='upper left',fontsize=9,frameon=False)
    fig.suptitle('CoolSiC-style trench JFET | OFF-state 2D electrostatic Green-operator model',
                 fontsize=19,fontweight='bold',color='#123b39')
    fig.text(0.5,0.025,'Assumed 4H-SiC, $N_D$=5×10$^{15}$ cm$^{-3}$, $V_G$=−15 V; fixed P+ bottom = 0.45 µm  |  $E_{crit}$=2.5 MV/cm over 0.20 µm; NOT avalanche BV',
             ha='center',fontsize=10,color='#52656d')
    fig.savefig(ROOT/'coolsic_jfet_green_2d_and_vbd.png',dpi=190)
    fig.savefig(ROOT/'coolsic_jfet_green_2d_and_vbd.svg')
    plt.close(fig)
    # standalone E plot
    fig,ax=plt.subplots(figsize=(8.6,10.4),constrained_layout=True)
    me=ax.pcolormesh(X,Z,Emasked,cmap=cmap,vmin=0,vmax=E_CRIT,shading='auto')
    ax.contour(X,Z,Emasked,levels=np.arange(0.2, E_CRIT,0.2),colors='white',linewidths=0.7,alpha=.82)
    label_region(ax,m,'OFF state 2D |E| (0.20 µm averaged)',is_field=True)
    cb=fig.colorbar(me,ax=ax,shrink=0.72)
    cb.set_label('Averaged electric-field magnitude (MV/cm)',fontsize=11)
    ax.text(-4.8,12.0,f'V$_S$ = 0 V  |  V$_G$ = {VG:.0f} V\nV$_D$ = {plotVD:.0f} V',fontsize=9,
            bbox=dict(boxstyle='round,pad=.4',facecolor='white',alpha=.91),color='#345',zorder=20)
    fig.savefig(ROOT/'coolsic_jfet_green_efield.png',dpi=190)
    plt.close(fig)
    # standalone potential
    fig,ax=plt.subplots(figsize=(8.6,10.4),constrained_layout=True)
    pe=ax.pcolormesh(X,Z,Vmasked,cmap='turbo',vmin=VG,vmax=plotVD,shading='auto')
    ax.contour(X,Z,np.ma.array(V,mask=m['fixed']),levels=np.linspace(0,plotVD,18),colors='white',linewidths=.6)
    label_region(ax,m,'OFF state potential V (2D)',is_field=True)
    fig.colorbar(pe,ax=ax,shrink=.72,label='Electrostatic potential (V)')
    fig.savefig(ROOT/'coolsic_jfet_potential.png',dpi=190)
    plt.close(fig)
    np.savez_compressed(ROOT/'coolsic_jfet_green_data.npz',x_um=x,z_um=z,potential_V=V,
                        Ex_MVcm=ex,Ez_MVcm=ez,E_raw_MVcm=E,E_avg_MVcm=np.where(valid,E_eff,np.nan),pplus_mask=m['pplus'],gate_mask=m['gate'],
                        source_mask=m['source'],drain_mask=m['drain'],Vd_V=plotVD,
                        width_um=BASE_W,width_sweep_um=ws,Vbd_estimate_V=vbs)
    # Same drain bias: compare three P+ collar widths at common E color scale
    fig, axs = plt.subplots(1,3,figsize=(14.5,7.3),constrained_layout=True)
    for ax,w in zip(axs,[0.20,0.50,0.80]):
        sol = next(s for s in solutions if np.isclose(s['width'],w))
        mm=sol['m']; vv=sol['vb'] + COMPARE_VD*sol['vd']
        _,_,ee,act=field(vv,mm)
        ee_avg,good=averaged_E(ee,act)
        mp=ax.pcolormesh(X,Z,np.ma.array(ee_avg,mask=~good),cmap=cmap,
                         vmin=0,vmax=E_CRIT,shading='auto')
        levels=np.arange(0.3,E_CRIT,0.3)
        ax.contour(X,Z,np.ma.array(ee_avg,mask=~good),levels=levels,colors='white',
                   linewidths=0.55,alpha=.75)
        label_region(ax,mm, f'P+ width = {w:.2f} µm', is_field=True)
        ax.text(0,10.8,f'Avg peak |E|: {np.max(ee_avg[good]):.2f} MV/cm',ha='center',
                color='white',fontsize=10,weight='bold',zorder=14)
    cbar=fig.colorbar(mp,ax=list(axs),pad=0.02,shrink=0.75,aspect=25)
    cbar.set_label('0.20 µm averaged |E| (MV/cm)',fontsize=11)
    fig.suptitle('P+ width changes 2D electric field | common $V_D$ = +800 V',
                 fontsize=17,fontweight='bold',color='#123b39')
    fig.savefig(ROOT/'coolsic_jfet_efield_width_comparison.png',dpi=180)
    plt.close(fig)



def main():
    solutions=[]
    for w in PPLUS_WIDTHS:
        m,vb,vd,resid=sparse_green_system(w)
        vbd,emax=get_critical_voltage(m,vb,vd)
        _,_,E,active=field(vb + vbd*vd,m)
        E_eff, valid=averaged_E(E,active)
        idx = np.nanargmax(np.where(valid,E_eff,np.nan))
        iz,ix=np.unravel_index(idx,E.shape)
        gap=2*TRENCH_CENTERS[1]-2*(TRENCH_HALF_WIDTH+w)
        print(f'w={w:.2f} um   Vcrit_Eavg={vbd:.1f} V  avg-peak={emax:.3f} MV/cm '
              f'location x={x[ix]:.2f}, z={z[iz]:.2f} um  gap={gap:.2f} um  residual={resid:.2e}',flush=True)
        solutions.append(dict(width=w,Vbd=vbd,m=m,vb=vb,vd=vd,peak=emax,loc_x=x[ix],loc_z=z[iz],gap=gap))
    with open(ROOT/'coolsic_jfet_width_sweep.csv','w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Pplus_width_um','Averaged_field_threshold_voltage_V','averaged_peak_field_MV_cm','x_of_peak_um','z_of_peak_um',
                         'channel_gap_um','relative_inverse_gap_vs_w0_20'])
        g0=solutions[0]['gap']
        for s in solutions:
            writer.writerow([s['width'],s['Vbd'],s['peak'],s['loc_x'],s['loc_z'],s['gap'],g0/s['gap']])
    make_figures(solutions)
    print('wrote figure, SVG, standalone maps, CSV and NPZ',flush=True)

if __name__ == '__main__':
    main()

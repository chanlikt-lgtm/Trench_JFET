#!/usr/bin/env python3
"""Normally-on 4H-SiC JFET; nonlinear Poisson solved via a discrete Green operator.

The P+ columns are SEMICONDUCTOR, not gate metal. The gate metal contacts
small strips on the OUTER faces of the P+ columns. Electron and hole carrier
screening is explicitly included in a Boltzmann approximation at VDS=0.

- BC: Vs=Vd=0; gate 0 V (on) or negative (pinch-off demonstration)
- Outside: dV/dn=0
- The nonlinear Green operator is (A+diag(kappa^2))^-1 per Newton step.
- This is an equilibrium/electrostatic calculation, NOT a drain-current or
  high-VDS breakdown simulation.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import splu

OUT=Path(__file__).resolve().parent
q=1.602176634e-19
EPS=8.8541878128e-12*9.7
VT=0.025852 # 300 K (V)
NI=8e-9 # cm^-3, illustrative 4H-SiC value at 300K
N_A=5e17
N_CH=1e16
N_DRIFT=5e15
N_PLUS=1e18

# u = electrostatic potential in volts, x,z in um
x=np.linspace(-3,3,121)
z=np.linspace(0,8,161)
X,Z=np.meshgrid(x,z)
nz,nx=X.shape
dx=x[1]-x[0]
dz=z[1]-z[0]

pcol=(np.abs(X)>=0.90-1e-8)&(np.abs(X)<=1.55+1e-8)&(Z>=0.55-1e-8)&(Z<=3.05+1e-8)
src_plus=(np.abs(X)<0.9+1e-8)&(Z<=0.55+1e-8)
drn_plus=(Z>=7.55-1e-8)
channel=(np.abs(X)<0.9-1e-8)&(Z>=0.55)&(Z<=3.05)
ND=np.full(X.shape,N_DRIFT,dtype=float)
ND[channel]=N_CH
ND[src_plus|drn_plus]=N_PLUS
ND[pcol]=0
NA=np.zeros(X.shape,dtype=float)
NA[pcol]=N_A

# Dirichlet conditions only on METAL contacts; semiconductor P+ interior
# remains part of the PDE domain and carries acceptors and holes.
source=(Z<=0.000001)&(np.abs(X)<=0.8+1e-8)
drain=(Z>=z[-1]-1e-8)
gate=(np.abs(np.abs(X)-1.55)<=dx*0.49)&(Z>=0.80-1e-8)&(Z<=1.60+1e-8)
assert (gate&pcol).sum()>0
fix=source|drain|gate

# A is the negative Laplacian with insulating (zero Neumann) side boundaries.
free=np.argwhere(~fix)
idx=-np.ones(X.shape,int)
idx[~fix]=np.arange(len(free))
NN=len(free)
A=lil_matrix((NN,NN),dtype=float)
# Because top/contact Dirichlet values depend on gate bias, create bBC each run.
bc_terms=[]
neighbors=[(-1,0,1/dz**2),(1,0,1/dz**2),(0,-1,1/dx**2),(0,1,1/dx**2)]
for k,(i,j) in enumerate(free):
    diagonal=0
    for di,dj,coef in neighbors:
        ii,jj=i+di,j+dj
        if not (0<=ii<nz and 0<=jj<nx):
            continue
        diagonal+=coef
        if fix[ii,jj]:
            bc_terms.append((k,ii,jj,coef))
        else:
            A[k,idx[ii,jj]]=-coef
    A[k,k]=diagonal
A=A.tocsr()

# q Nd / eps with cm^-3 to um^-2 conversion: (cm^-3 -> m^-3)* (um^2 -> m^2)
coef_rho=q*1e-6/EPS
# Quasi-Fermi potentials: electrons referenced to source (0 V), holes in
# P+ tied to gate Vg, allowing a reverse-biased p-n junction.

def carrier(phi,vg):
    fp=np.where(pcol,vg,0.)
    ne=NI*np.exp(np.clip(phi/VT,-120,120))
    hp=NI*np.exp(np.clip((fp-phi)/VT,-120,120))
    return ne,hp


def solve(vg,guess=None):
    # Neutral bulk contact potentials include the built-in semiconductor
    # voltage; Vg is a terminal VOLTAGE, not the P+ electrostatic potential.
    phi_nplus=VT*np.arcsinh(N_PLUS/(2*NI))
    phi_pplus=vg-VT*np.arcsinh(N_A/(2*NI))
    values=np.zeros(X.shape)
    values[source|drain]=phi_nplus
    values[gate]=phi_pplus

    boundary=np.zeros(NN)
    for k,ii,jj,coef in bc_terms:
        boundary[k]+=coef*values[ii,jj]

    net=ND-NA
    initial=VT*np.arcsinh(np.clip(net/(2*NI),-1e100,1e100))
    initial[pcol]=phi_pplus
    if guess is not None:
        initial=guess.copy()
    initial[fix]=values[fix]
    phi=initial[~fix].copy()

    def residual(U):
        full=values.copy()
        full[~fix]=U
        n,p=carrier(full,vg)
        F=A@U-boundary-coef_rho*(net[~fix]+p[~fix]-n[~fix])
        return F,n[~fix],p[~fix]

    logs=[]
    for t in range(100):
        F,ne,hp=residual(phi)
        r=np.linalg.norm(F)/np.sqrt(NN)
        logs.append(r)
        if t%5==0:
            print(f'vg={vg:+g} iteration {t}: residual RMS {r:.5g} V/um^2',flush=True)
        if r<5e-7:
            break
        Jac=A+diags(coef_rho*(ne+hp)/VT,format='csr')
        lu=splu(Jac.tocsc())
        d=lu.solve(-F)
        # Damped Newton: the device has sub-grid p+ Debye length.
        alpha=min(1.,0.35/max(np.max(np.abs(d)),1e-12))
        for _ in range(17):
            trial=phi+alpha*d
            ft,_,_=residual(trial)
            if np.linalg.norm(ft) < np.linalg.norm(F):
                phi=trial
                break
            alpha*=0.5
        else:
            raise RuntimeError(f'Newton line search failed at t={t}, rms={r}')
    else:
        raise RuntimeError(f'Newton did not converge, rms={logs[-1]}')
    result=values.copy()
    result[~fix]=phi
    ne,hp=carrier(result,vg)
    dphidz,dphidx=np.gradient(result,dz,dx,edge_order=2)
    EX=-dphidx*0.01 # V/um * .01 = MV/cm
    EZ=-dphidz*0.01
    EMAG=np.hypot(EX,EZ)
    return dict(phi=result,n=ne,p=hp,Ex=EX,Ez=EZ,E=EMAG,iterations=t+1,rms=r)


# ONLY the zero-gate, zero-drain equilibrium is solved here. This is the
# electrostatic normally-ON state, with a neutral n-type central channel.
print("Solving VGS = VDS = 0 V normally-ON equilibrium:",flush=True)
on=solve(0)
phi_nplus=VT*np.arcsinh(N_PLUS/(2*NI))
Vbi=VT*np.log(N_A*N_CH/NI**2)
Nd=N_CH*1e6
Na=N_A*1e6
Wn=np.sqrt(2*EPS*Vbi/q*Na/(Nd*(Na+Nd)))*1e6
Wp=Wn*N_CH/N_A
Wneutral=max(0.,1.8-2*Wn)
E_junc=(q*Nd*(Wn*1e-6)/EPS)/1e8

print(f"Vbi = {Vbi:.3f} V; 1D Wn = {Wn:.3f} um; Wp = {Wp:.4f} um")
print(f"Approximate neutral channel width = {Wneutral:.3f} um")
print(f"1D interface field = {E_junc:.3f} MV/cm; 2D computed peak = {on['E'].max():.3f} MV/cm")

from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12})
fig=plt.figure(figsize=(14,8.9),facecolor='white')
gs=fig.add_gridspec(1,3,left=.06,right=.97,top=.855,bottom=.155,wspace=.30)
axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
fig.suptitle('Normally-ON vertical 4H-SiC JFET  |  gate METAL on outer face of P+ SiC columns',
             fontsize=15,fontweight='bold',y=.98)
fig.text(.5,.928,'2D nonlinear Poisson solved by the discrete screened Green operator  ·  VGS = 0 V   ·   VDS = 0 V (equilibrium)',
         ha='center',fontsize=11,color='#45556c')

for ax,title in zip(axs,['A  Device / boundary conditions','B  2D electric-field magnitude','C  2D electrostatic potential']):
    ax.set_title(title,pad=11,fontweight='bold',color='#15344f')
    ax.set_xlim(-3.0,3.0)
    ax.set_ylim(8.1,-.13)
    ax.set_xlabel('Lateral x (µm)')
    ax.set_ylabel('Depth z (µm)')
    ax.set_aspect('equal')
    ax.set_xticks([-3,-1.5,0,1.5,3])
    ax.tick_params(labelsize=9)

ax=axs[0]
ax.set_facecolor('#f4e6b9')
for m,c in ((src_plus,'#f8c342'),(drn_plus,'#efb23c'),(pcol,'#e59dab')):
    ax.contourf(X,Z,m.astype(float),levels=[.5,1.5],colors=[c])
# ONLY outer-side gray rectangles are metal gates. P+ remains pink semiconductor.
for xx in (-1.88,1.55):
    ax.add_patch(patches.Rectangle((xx,.80),.33,.8,facecolor='#48535f',edgecolor='#263447',lw=.9,zorder=6))
ax.add_patch(patches.Rectangle((-.8,-.10),1.6,.16,facecolor='#48535f',zorder=6))
ax.add_patch(patches.Rectangle((-3,7.97),6,.16,facecolor='#48535f',zorder=6))
ax.text(0,.31,'n+ source',ha='center',va='center',fontsize=9,weight='bold')
ax.text(0,1.65,'n-channel\nOPEN',ha='center',va='center',fontsize=11,weight='bold',color='#123f6a')
for xx in (-1.22,1.22):
    ax.text(xx,2.17,'P+\nSiC',ha='center',fontsize=9,weight='bold',color='#73313c')
ax.text(0,5.2,'n− SiC drift',ha='center',fontsize=11,weight='bold',color='#53491c')
ax.annotate('Gate metal\nVGS = 0',xy=(-1.72,1.25),xytext=(-2.85,3.15),fontsize=9,
             arrowprops=dict(arrowstyle='->',color='#35475f',lw=1.3),color='#263851')
ax.annotate('Gate metal\nVGS = 0',xy=(1.72,1.25),xytext=(1.80,3.15),fontsize=9,
             arrowprops=dict(arrowstyle='->',color='#35475f',lw=1.3),color='#263851')

cmap=LinearSegmentedColormap.from_list('greens', ['#053426','#116343','#16a75b','#8add59','#f0fb66'])
field=on['E']
# For contour legibility the color upper bound follows the calculated 2D maximum.
maxE=float(field.max())
upper=max(.05,np.ceil(maxE*20)/20)
im=axs[1].pcolormesh(X,Z,field,cmap=cmap,vmin=0,vmax=upper,shading='auto',rasterized=True)
axs[1].contour(X,Z,field,levels=np.linspace(.025,upper,13),colors='white',linewidths=.55,alpha=.83)
axs[1].contour(X,Z,pcol.astype(float),levels=[.5],colors='#ffd4ab',linewidths=1.2)
cb=fig.colorbar(im,ax=axs[1],orientation='horizontal',fraction=.043,pad=.095,aspect=26)
cb.set_label('Magnitude |E| (MV/cm)',fontsize=10)

phi=on['phi']-phi_nplus
im2=axs[2].pcolormesh(X,Z,phi,cmap='coolwarm',vmin=-3.25,vmax=.1,shading='auto',rasterized=True)
cs=axs[2].contour(X,Z,phi,levels=np.arange(-3.2,.01,.20),colors='#254b60',linewidths=.50,alpha=.8)
axs[2].contour(X,Z,pcol.astype(float),levels=[.5],colors='#ffe19a',linewidths=1.1)
cb2=fig.colorbar(im2,ax=axs[2],orientation='horizontal',fraction=.043,pad=.095,aspect=26)
cb2.set_label('Electrostatic φ − φsource (V)',fontsize=10)

fig.text(.5,.07,'P+ columns: NA = 5×10¹⁷ cm⁻³  ·  n channel: ND = 1×10¹⁶ cm⁻³  ·  4H-SiC εr = 9.7',
         ha='center',fontsize=10,color='#29465a')
fig.text(.5,.046,'Metal BC: source = 0 V, drain = 0 V, gates = 0 V. P+/n interfaces: potential + normal displacement continuous.',
         ha='center',fontsize=9,color='#486176')
fig.savefig(OUT/'normal_on_pcolumn_green_2d.png',dpi=200,facecolor='white')
fig.savefig(OUT/'normal_on_pcolumn_green_2d.svg')
plt.close(fig)

# Carrier density across the channel, a direct check of normally-ON state.
iz=int(np.argmin(abs(z-1.70)))
fig2,ax=plt.subplots(figsize=(8.3,4.2))
ax.semilogy(x,on['n'][iz],color='#136346',lw=2.5)
ax.axvspan(-.9,.9,color='#9ddef4',alpha=.22,label='n-type channel')
ax.axhline(N_CH,color='#697680',ls='--',lw=1.1,label='ND channel')
ax.axvline(-.9,color='#af6f72',ls=':',lw=1.1)
ax.axvline(.9,color='#af6f72',ls=':',lw=1.1)
ax.set_xlim(-2.1,2.1)
ax.set_ylim(1e0,2e17)
ax.set_title('Normally ON at VGS = 0: n-channel retains mobile electrons')
ax.set_xlabel('Lateral x (µm), at z = 1.70 µm')
ax.set_ylabel('Electron density n (cm⁻³)')
ax.legend(loc='upper right')
ax.grid(alpha=.26,which='both')
fig2.tight_layout()
fig2.savefig(OUT/'normal_on_channel_carrier_density.png',dpi=190)
plt.close(fig2)

np.savez_compressed(OUT/'normal_on_pcolumn_green_data.npz',x_um=x,z_um=z,
                    dopant_ND_cm3=ND,dopant_NA_cm3=NA,Pplus_columns=pcol,
                    gate_metal_contact=gate,potential_rel_source_V=phi,
                    Ex_MV_cm=on['Ex'],Ez_MV_cm=on['Ez'],E_MV_cm=on['E'],
                    electron_n_cm3=on['n'],holes_p_cm3=on['p'])
with (OUT/'normal_on_green_results.txt').open('w') as f:
    f.write('Normally ON SiC JFET / nonlinear Poisson / discrete Green operator\n')
    f.write(f'P+ doping={N_A:.3e} cm^-3; n channel doping={N_CH:.3e} cm^-3\n')
    f.write('Gate metal contacts ONLY OUTER P+ FACES, not whole columns.\n')
    f.write(f'1D abrupt-junction Wn={Wn:.4f} um; Wp={Wp:.5f} um; open core={Wneutral:.4f} um\n')
    f.write(f'Computed 2D peak E={maxE:.4f} MV/cm (mesh-dependent near junction corners)\n')
    f.write('VDS=0: equilibrium electrostatics; neither Id-Vg nor breakdown simulated.\n')
print('OUTPUT:',OUT,'peak E',maxE,'open width from 1D',Wneutral)

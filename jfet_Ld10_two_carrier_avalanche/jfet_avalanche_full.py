#!/usr/bin/env python3
"""Exploratory 2D silicon JFET drift diffusion with electron/hole impact ionization.

NOT a calibrated breakdown prediction. Explicit contact geometry, Poisson-Green
(operator inverse), Scharfetter-Gummel electron/hole flux and self-consistent
source G_II=(alpha_n |Jn|+alpha_p |Jp|)/q. For breakdown testing also
estimate the small-signal two-carrier avalanche feedback gain.

All source/drain N+ lateral depths default to 0.2 um, an assumed parameter.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from dataclasses import dataclass

Q=1.602176634e-19
EPS=11.7*8.8541878128e-14 # F/cm
VT=0.025852 # V at ~300K
NI=1.e10 # cm^-3
N0=1.e16 # cm^-3 density normalization
C=Q*N0/EPS*1.e-8 # V/um^2
MUN=1350. # cm^2/Vs, low-field assumption
MUP=480.
TAUN=1.e-6
TAUP=1.e-6
A_N,B_N=7.03e5,1.231e6 # van Overstraeten-de Man, alpha_n cm^-1
A_P1,B_P1=1.582e6,2.036e6 # <= 4e5 V/cm
A_P2,B_P2=6.71e5,1.693e6 # > 4e5 V/cm


def bern(z):
    z=np.asarray(z,dtype=float)
    out=np.empty_like(z)
    a=np.abs(z)<1e-4; positive=z>40; negative=z< -40
    mid=~(a|positive|negative)
    out[a]=1-z[a]/2+z[a]*z[a]/12-z[a]**4/720
    out[positive]=z[positive]*np.exp(-z[positive])/(1.-np.exp(-z[positive]))
    out[negative]=-z[negative]/(1.-np.exp(z[negative]))
    out[mid]=z[mid]/np.expm1(z[mid])
    return out


def contact_properties(doping_norm):
    pot=VT*np.arcsinh(doping_norm/(2*NI/N0))
    n=(NI/N0)*np.exp(pot/VT)
    p=(NI/N0)*np.exp(-pot/VT)
    return pot,n,p


def ionization_coeff(E):
    """Electron and hole alpha(E) in cm-1 from van Overstraeten-de Man.
    Outside 1.75e5 .. 6.4e5 V/cm values are extrapolated, NOT validated.
    """
    E=np.maximum(np.asarray(E),1.0)
    an=A_N*np.exp(-B_N/E)
    ap=np.where(E<=4e5,A_P1*np.exp(-B_P1/E),A_P2*np.exp(-B_P2/E))
    return an,ap


class AvalancheJFET:
    def __init__(self,a=.5,NA=1e17,Ld=10.,dx=.1,dy=.1,ws=.2,wd=.2,
                 mu_n=MUN,mu_p=MUP,tau_n=TAUN,tau_p=TAUP):
        self.a=a; self.NA=NA; self.Ld=Ld
        self.dx=dx;self.dy=dy;self.ws=ws;self.wd=wd
        self.mu_n=mu_n; self.mu_p=mu_p;self.tau_n=tau_n;self.tau_p=tau_p
        self.L=1.; self.H=2.; self.drainx=1.+Ld
        assert 0<a<1 and Ld>0
        self.Nx=round((ws+self.drainx+wd)/dx)
        self.Ny=round(2/dy)
        assert abs(self.Nx*dx-(ws+self.drainx+wd))<1e-9
        assert abs(self.Ny*dy-2)<1e-9
        self.x=-ws+dx*(np.arange(self.Nx)+.5)
        self.y=-1.+dy*(np.arange(self.Ny)+.5)
        self.X,self.Y=np.meshgrid(self.x,self.y)
        self.N=self.Nx*self.Ny
        nplus=(self.X<0)|(self.X>=self.drainx)
        pplus=(self.X>=0)&(self.X<1.)&(np.abs(self.Y)>=1-a)
        self.nplus=nplus
        self.pplus=pplus
        self.Nd=np.where(nplus,1e18,np.where(pplus,-NA,1e16))
        self.D=(self.Nd/N0).ravel()
        self.peq=contact_properties(-NA/N0)
        self.neq=contact_properties(100.)
        base=np.arange(self.N).reshape(self.Ny,self.Nx)
        self.i_x=base[:,:-1].ravel(); self.j_x=base[:,1:].ravel()
        self.i_y=base[:-1,:].ravel(); self.j_y=base[1:,:].ravel()
        self.i=np.concatenate((self.i_x,self.i_y))
        self.j=np.concatenate((self.j_x,self.j_y))
        self.ce=np.concatenate((np.full(len(self.i_x),1/dx**2),np.full(len(self.i_y),1/dy**2)))
        self.d_cm=np.concatenate((np.full(len(self.i_x),dx*1e-4),np.full(len(self.i_y),dy*1e-4)))
        self.nedges=len(self.i)
        self.contacts={}
        self.contacts['source']=(base[:,0],2/dx**2, self.neq,0.)
        self.contacts['drain']=(base[:,-1],2/dx**2, self.neq,0.)
        gcols=np.where((self.x>=0)&(self.x<1))[0]
        self.contacts['gtop']=(base[-1,gcols],2/dy**2,self.peq,0.)
        self.contacts['gbot']=(base[0,gcols],2/dy**2,self.peq,0.)
        self.contact_indices=np.concatenate([c[0] for c in self.contacts.values()])
        diag=np.zeros(self.N)
        np.add.at(diag,self.i,self.ce);np.add.at(diag,self.j,self.ce)
        for idx,coef,pr,v in self.contacts.values(): diag[idx]+=coef
        self.A=sp.coo_matrix((np.concatenate((-self.ce,-self.ce,diag)),
             (np.concatenate((self.i,self.j,np.arange(self.N))),
              np.concatenate((self.j,self.i,np.arange(self.N))))),shape=(self.N,self.N)).tocsc()
        self.Alu=sla.splu(self.A)
        self.psi=None;self.n=None;self.p=None
        self.Vg=0.;self.Vd=0.
        self.generation=None
        self._contact_rhs={}

    def contacts_rhs(self,Vs=0.,Vd=0.,Vg=-4.):
        b=np.zeros(self.N)
        for tag,(idx,coef,props,unused) in self.contacts.items():
            vv=Vs if tag=='source' else Vd if tag=='drain' else Vg
            b[idx]+=coef*(props[0]+vv)
        return b

    def eq_start(self):
        psi=VT*np.arcsinh(self.D/(2*NI/N0))
        b=self.contacts_rhs(0.,0.,0.)
        for it in range(90):
            u=np.clip(psi/VT,-100,100)
            n=(NI/N0)*np.exp(u); p=(NI/N0)*np.exp(-u)
            r=self.A@psi-b-C*(self.D+p-n)
            err=np.max(np.abs(r))
            if err<1e-7:break
            jac=self.A+sp.diags(C/VT*(n+p))
            step=sla.spsolve(jac,-r)
            sm=np.max(np.abs(step))
            if sm>.15:step*=.15/sm
            aa=1.
            while aa>1e-5:
                pn=psi+aa*step
                v=np.clip(pn/VT,-100,100)
                rn=self.A@pn-b-C*(self.D+(NI/N0)*np.exp(-v)-(NI/N0)*np.exp(v))
                if np.max(np.abs(rn))<=err*(1+1e-8):break
                aa/=2.
            psi+=aa*step
        self.psi=psi;self.n=(NI/N0)*np.exp(np.clip(psi/VT,-100,100));self.p=(NI/N0)*np.exp(np.clip(-psi/VT,-100,100))
        self.b=b
        self.generation=np.zeros(self.N)
        return it,err

    def build_transport(self,psi,electron):
        """M matrix, contact rhs, and face flux coeffs for SG transport.
        M*carrier = rhs_contact + 1e-8/(VT*N0) * (G-R), with Mu in M.
        """
        dv=(psi[self.j]-psi[self.i])/VT
        bp=bern(dv);bm=bern(-dv)
        mu=self.mu_n if electron else self.mu_p
        c=self.ce*mu
        if electron:
            di=c*bm;dj=c*bp;ij=-c*bp;ji=-c*bm
        else:
            di=c*bp;dj=c*bm;ij=-c*bm;ji=-c*bp
        diag=np.bincount(self.i,weights=di,minlength=self.N)+np.bincount(self.j,weights=dj,minlength=self.N)
        rhs=np.zeros(self.N)
        for tag,(idx,coef,props,unused) in self.contacts.items():
            v=0. if tag=='source' else self.Vd if tag=='drain' else self.Vg
            boundary=props[0]+v
            d=(boundary-psi[idx])/VT
            Bp=bern(d); Bm=bern(-d)
            if electron:
                diag[idx]+=mu*coef*Bm
                rhs[idx]+=mu*coef*Bp*props[1]
            else:
                diag[idx]+=mu*coef*Bp
                rhs[idx]+=mu*coef*Bm*props[2]
        M=sp.coo_matrix((np.concatenate((ij,ji,diag)),
            (np.concatenate((self.i,self.j,np.arange(self.N))),
             np.concatenate((self.j,self.i,np.arange(self.N))))), shape=(self.N,self.N)).tocsc()
        return M,rhs,bp,bm

    def flux_and_generation(self,psi,n,p, bp=None,bm=None):
        """Compute current-based generation G_II in cm-3/s, and field."""
        if bp is None:
            dv=(psi[self.j]-psi[self.i])/VT
            bp=bern(dv);bm=bern(-dv)
        # conventional currents per q, particle/(cm^2 s), faces oriented +x,+y
        Jn=self.mu_n*VT*N0/self.d_cm*(n[self.j]*bp-n[self.i]*bm)
        Jp=self.mu_p*VT*N0/self.d_cm*(p[self.i]*bp-p[self.j]*bm)
        nfx=Jn[:len(self.i_x)].reshape(self.Ny,self.Nx-1)
        nfy=Jn[len(self.i_x):].reshape(self.Ny-1,self.Nx)
        pfx=Jp[:len(self.i_x)].reshape(self.Ny,self.Nx-1)
        pfy=Jp[len(self.i_x):].reshape(self.Ny-1,self.Nx)
        def face_to_node(fx,fy):
            xx=np.empty((self.Ny,self.Nx)); yy=np.empty_like(xx)
            xx[:,1:-1]=(fx[:,1:]+fx[:,:-1])*.5
            xx[:,0]=fx[:,0];xx[:,-1]=fx[:,-1]
            yy[1:-1,:]=(fy[1:,:]+fy[:-1,:])*.5
            yy[0,:]=fy[0,:];yy[-1,:]=fy[-1,:]
            return np.sqrt(xx*xx+yy*yy)
        Jna=face_to_node(nfx,nfy)
        Jpa=face_to_node(pfx,pfy)
        pa=psi.reshape(self.Ny,self.Nx)
        ex=np.gradient(-pa,self.dx,axis=1)*1e4
        ey=np.gradient(-pa,self.dy,axis=0)*1e4
        E=np.sqrt(ex*ex+ey*ey)
        an,ap=ionization_coeff(E)
        generation=an*Jna+ap*Jpa
        return generation.ravel(),float(E.max()),E

    def srh_recombination(self,n,p):
        # cm-3 s-1; n,p are normalized densities
        ni=NI/N0
        num=n*p-ni*ni
        den=self.tau_p*(n+ni)+self.tau_n*(p+ni)
        return N0*num/np.maximum(den,1e-90)

    def poisson_update(self,psi_ref,n_ref,p_ref,b,maxit=26):
        psi=psi_ref.copy()
        for k in range(maxit):
            dv=np.clip((psi-psi_ref)/VT,-70,70)
            n=n_ref*np.exp(dv);p=p_ref*np.exp(-dv)
            r=self.A@psi-b-C*(self.D+p-n)
            err=np.max(np.abs(r))
            if err<2e-7:break
            jac=self.A+sp.diags(C/VT*(n+p))
            step=sla.spsolve(jac,-r)
            max_step=np.max(np.abs(step))
            if max_step>.15:step*=.15/max_step
            alpha=1.
            while alpha>2e-5:
                pt=psi+alpha*step
                dt=np.clip((pt-psi_ref)/VT,-70,70)
                nr=n_ref*np.exp(dt);pr=p_ref*np.exp(-dt)
                er=np.max(np.abs(self.A@pt-b-C*(self.D+pr-nr)))
                if er <= err*(1+1e-8):break
                alpha/=2
            psi+=alpha*step
        dv=np.clip((psi-psi_ref)/VT,-70,70)
        return psi,n_ref*np.exp(dv),p_ref*np.exp(-dv),err

    def solve(self,Vd,Vg=-4.,avalanche=True,maxit=90,verbose=False,
              damp=.5,tol=3e-5):
        if self.psi is None:self.eq_start()
        b=self.contacts_rhs(0,Vd,Vg)
        if self.b is not None:
            self.psi=self.psi+self.Alu.solve(b-self.b)
        self.Vd=Vd;self.Vg=Vg
        psi=self.psi.copy();n=self.n.copy();p=self.p.copy()
        G=self.generation.copy() if avalanche else np.zeros(self.N)
        srh0=np.zeros(self.N)
        stat={'converged':False,'Vd':float(Vd),'Vg':float(Vg),'avalanche':avalanche}
        for it in range(maxit):
            mn,rhsn,bp,bm=self.build_transport(psi,True)
            mp,rhsp,_,_=self.build_transport(psi,False)
            srh=self.srh_recombination(n,p)
            source=np.clip((G-srh)/N0*(1e-8/VT),-1e20,1e20)
            try:
                nsol=sla.spsolve(mn,rhsn+source)
                psol=sla.spsolve(mp,rhsp+source)
            except Exception:
                break
            if (not np.all(np.isfinite(nsol))) or (not np.all(np.isfinite(psol))) or min(nsol.min(),psol.min()) < -1e-6:
                break
            nsol=np.maximum(nsol,1e-70);psol=np.maximum(psol,1e-70)
            psi2,n2,p2,pres=self.poisson_update(psi,nsol,psol,b)
            if not np.all(np.isfinite(psi2)): break
            delta=np.max(np.abs(psi2-psi))
            psi=(1-damp)*psi+damp*psi2
            n=(1-damp)*n+damp*n2
            p=(1-damp)*p+damp*p2
            if avalanche:
                gen_now,Emax,E=self.flux_and_generation(psi,n,p)
                G=(1-damp)*G+damp*gen_now
            else:
                gen_now=np.zeros_like(G)
                Emax=np.nan
            if verbose and (it%10==0 or delta<tol):
                print('vd',Vd,'it',it,'delta',delta,'genmax',G.max(),'maxfield',Emax,flush=True)
            if delta<tol and (not avalanche or np.max(np.abs(gen_now-G))/max(G.max(),1e10)<.01):
                stat['converged']=True
                break
        stat.update({'iters':it+1,'delta':float(delta),'Gmax':float(G.max()),
                     'Poisson_res':float(np.max(np.abs(self.A@psi-b-C*(self.D+p-n))))})
        self.psi=psi;self.n=n;self.p=p;self.b=b;self.generation=G
        self.last=stat
        return stat

    def feedback_gain(self,psi=None,steps=20,seed=None):
        """Two-carrier, frozen-field small-signal avalanche gain operator norm.
        Not a self-consistent breakdown voltage or full eigenvalue proof.
        """
        if psi is None:psi=self.psi
        mn,_,bp,bm=self.build_transport(psi,True)
        mp,_,_,_=self.build_transport(psi,False)
        lun=sla.splu(mn); lup=sla.splu(mp)
        if seed is None:
            seed=np.ones(self.N)
        f=np.maximum(seed,1e-40)
        f/=f.sum()
        growth=0.
        for it in range(steps):
            src=f*(1e-8/VT)
            n=np.maximum(lun.solve(src),0.)
            p=np.maximum(lup.solve(src),0.)
            G,Em,E=self.flux_and_generation(psi,n,p,bp,bm)
            # source: one 'carrier per density N0' / cm^3/s (scale cancels)
            G=G/N0
            growth=G.sum()
            if not np.isfinite(growth) or growth<=0: return 0.,Em
            f=G/growth
        return float(growth),Em

if __name__=='__main__':
    import time, argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--dx',type=float,default=.1)
    ap.add_argument('--dy',type=float,default=.1)
    ap.add_argument('--Ld',type=float,default=10.)
    ap.add_argument('--a',type=float,default=.5)
    ap.add_argument('--NA',type=float,default=1e17)
    ap.add_argument('--avalanche',action='store_true')
    args=ap.parse_args()
    t=time.time()
    m=AvalancheJFET(a=args.a,NA=args.NA,Ld=args.Ld,dx=args.dx,dy=args.dy)
    print('grid',m.Nx,m.Ny,'equilibrium',m.eq_start(),flush=True)
    for v in [-1.,-2.,-3.,-4.]:
        status=m.solve(0.,v,avalanche=False,maxit=130)
        print('gate',v,status,'t',time.time()-t,flush=True)
    for v in [1,5,10,20,30,40,50,60,80,100]:
        status=m.solve(v,-4.,avalanche=args.avalanche,maxit=110)
        gain,ef=m.feedback_gain(steps=12)
        print('drain',v,status,'gain',gain,'Emax',ef,'t',time.time()-t,flush=True)
        if not status['converged']:
            print('not converged, stopping');break

#!/usr/bin/env python3
"""Coupled Newton drift diffusion solver for AvalancheJFET.
Unknowns ψ [V], ln(n/N0), ln(p/N0), with Scharfetter-Gummel face flux.
G_II lagged in inner Newton, recomputed/damped in outer avalanche iteration.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from jfet_avalanche_full import AvalancheJFET,VT,C,NI,N0,bern,Q


def bernprime(z):
    z=np.asarray(z,dtype=float)
    out=np.empty_like(z)
    a=np.abs(z)<1e-3; positive=z>40;negative=z< -40
    mid=~(a|positive|negative)
    out[a]=-0.5+z[a]/6-z[a]**3/180+z[a]**5/5040
    out[positive]=(np.exp(-z[positive])*(1-z[positive]-np.exp(-z[positive]))/((1-np.exp(-z[positive]))**2))
    out[negative]=-1.+np.exp(z[negative])*(z[negative]+1-np.exp(z[negative]))/((1-np.exp(z[negative]))**2)
    zz=z[mid]
    e=np.exp(zz)
    out[mid]=(e-1-zz*e)/np.expm1(zz)**2
    return out

class CoupledNewton:
    def __init__(self,model):
        self.m=model
        self.gamma=1e-8/(VT*N0)
        self.nsize=model.N
        self.nn=model.N
        self.A=model.A.tocsr()
        self.ones=np.ones(model.N)
        self.rsc=0
        self.last=None

    def residual_jac(self,Z,b,G=None,jac=True):
        m=self.m
        N=m.N
        psi=Z[:N]
        u=np.clip(Z[N:2*N],-40,80)
        v=np.clip(Z[2*N:],-40,80)
        n=np.exp(u);p=np.exp(v)
        r_psi=m.A@psi - b - C*(m.D+p-n)
        z=(psi[m.j]-psi[m.i])/VT
        bp=bern(z); bm=bern(-z)
        d_bp=bernprime(z); d_bm=bernprime(-z)
        mu_n_c=m.ce*m.mu_n
        mu_p_c=m.ce*m.mu_p
        fn=mu_n_c*(n[m.i]*bm-n[m.j]*bp)
        fp=mu_p_c*(p[m.i]*bp-p[m.j]*bm)
        r_n=np.bincount(m.i,weights=fn,minlength=N)-np.bincount(m.j,weights=fn,minlength=N)
        r_p=np.bincount(m.i,weights=fp,minlength=N)-np.bincount(m.j,weights=fp,minlength=N)
        # Jacobian blocks: each edge makes 4 entries per block
        if jac:
            e_uu_i=mu_n_c*n[m.i]*bm
            e_uu_j=mu_n_c*n[m.j]*bp
            e_pp_i=mu_p_c*p[m.i]*bp
            e_pp_j=mu_p_c*p[m.j]*bm
            e_psi_n=mu_n_c*(n[m.i]*d_bm+n[m.j]*d_bp)/VT
            e_psi_p=-mu_p_c*(p[m.i]*d_bp+p[m.j]*d_bm)/VT
            row=np.concatenate([m.i,m.i,m.j,m.j])
            col=np.concatenate([m.i,m.j,m.i,m.j])
            data_n_u=np.concatenate([e_uu_i,-e_uu_j,-e_uu_i,e_uu_j])
            data_p_v=np.concatenate([e_pp_i,-e_pp_j,-e_pp_i,e_pp_j])
            data_n_psi=np.concatenate([e_psi_n,-e_psi_n,-e_psi_n,e_psi_n])
            data_p_psi=np.concatenate([e_psi_p,-e_psi_p,-e_psi_p,e_psi_p])
            rowc=[];nd_u=[];pd_v=[];nd_psi=[];pd_psi=[]
        for tag,(idx,coef,props,unused) in m.contacts.items():
            applied=0. if tag=='source' else m.Vd if tag=='drain' else m.Vg
            qpot=props[0]+applied
            zb=(qpot-psi[idx])/VT
            Bp=bern(zb); Bm=bern(-zb)
            r_n[idx]+=coef*m.mu_n*(n[idx]*Bm-props[1]*Bp)
            r_p[idx]+=coef*m.mu_p*(p[idx]*Bp-props[2]*Bm)
            if jac:
                dp=bernprime(zb);dm=bernprime(-zb)
                rowc.extend(idx)
                nd_u.extend(coef*m.mu_n*n[idx]*Bm)
                pd_v.extend(coef*m.mu_p*p[idx]*Bp)
                nd_psi.extend(coef*m.mu_n*(n[idx]*dm+props[1]*dp)/VT)
                pd_psi.extend(-coef*m.mu_p*(p[idx]*dp+props[2]*dm)/VT)
        if G is not None:
            r_n-=self.gamma*G
            r_p-=self.gamma*G
        # SRH optional based on m tau; use lagged R in Newton or disabled
        if jac:
            row2=np.concatenate([row,np.asarray(rowc,dtype=int)])
            col2=np.concatenate([col,np.asarray(rowc,dtype=int)])
            def blk(v,e):
                return sp.coo_matrix((np.concatenate([v,np.asarray(e)]),(row2,col2)),shape=(N,N)).tocsr()
            Jnu=blk(data_n_u,nd_u)
            Jpv=blk(data_p_v,pd_v)
            Jnpsi=blk(data_n_psi,nd_psi)
            Jppsi=blk(data_p_psi,pd_psi)
            dn=np.maximum(np.abs(Jnu.diagonal()),1e-2)
            dp=np.maximum(np.abs(Jpv.diagonal()),1e-2)
            # row equilibration; analytic Jacob to unscaled residual with fixed weights for each step
            Sn=sp.diags(1/dn)
            Sp=sp.diags(1/dp)
            # for robust balancing, Poisson block unchanged (V units)
            J=sp.bmat([[self.A,sp.diags(C*n),sp.diags(-C*p)],
                       [Sn@Jnpsi,Sn@Jnu,None],
                       [Sp@Jppsi,None,Sp@Jpv]],format='csc')
            r=np.concatenate([r_psi,r_n/dn,r_p/dp])
            return r,J,(n,p), (dn,dp)
        else:
            return r_psi,r_n,r_p,n,p

    def newton(self,Z,b,G=None,maxit=55,tol=2e-5,verbose=False):
        N=self.m.N
        Z=Z.copy()
        converged=False
        history=[]
        for it in range(maxit):
            r,J,(_, _),(dn,dp)=self.residual_jac(Z,b,G,jac=True)
            norm=np.max(np.abs(r))
            history.append(norm)
            if verbose and (it%5==0 or norm<tol):print('Newton',it,'norm',norm,'psimax',Z[:N].max(),flush=True)
            if norm<tol:
                converged=True
                break
            try:
                step=sla.spsolve(J,-r)
            except Exception as exc:
                print('Newton solver failed',exc,flush=True)
                break
            if not np.all(np.isfinite(step)):break
            alpha=1.0
            maxpsi=np.max(np.abs(step[:N])); maxlog=max(np.max(abs(step[N:2*N])),np.max(abs(step[2*N:])))
            if maxpsi>1.0:alpha=min(alpha,1.0/maxpsi)
            if maxlog>4.0:alpha=min(alpha,4.0/maxlog)
            # Weighted residual measured with fixed row scales
            r_psi,r_n,r_p,_,_=self.residual_jac(Z,b,G,jac=False)
            norm_current=max(np.max(abs(r_psi)),np.max(abs(r_n/dn)),np.max(abs(r_p/dp)))
            for _ in range(21):
                Znew=Z+alpha*step
                # prevent massive exponentials
                Znew[N:]=np.clip(Znew[N:],-40,80)
                rpsi,rn,rp,_,_=self.residual_jac(Znew,b,G,jac=False)
                norm_new=max(np.max(abs(rpsi)),np.max(abs(rn/dn)),np.max(abs(rp/dp)))
                if np.isfinite(norm_new) and norm_new < norm_current*(1-1e-4*alpha)+1e-12:
                    break
                alpha*=.5
            if alpha<1e-7:break
            Z=Znew
        return Z, {'newton_converged':converged, 'iters':it+1,'residual':float(history[-1]), 'history':history}

    def state0(self):
        m=self.m
        if m.psi is None:m.eq_start()
        return np.concatenate([m.psi,np.log(np.maximum(m.n,1e-65)),np.log(np.maximum(m.p,1e-65))])

    def run(self,Vd,Vg=-4.,Z=None,G=None,avalanche=False,outerit=25,tol=1e-3,verbose=False):
        m=self.m
        if Z is None:Z=self.state0()
        m.Vd=Vd;m.Vg=Vg
        b=m.contacts_rhs(0,Vd,Vg)
        if G is None:G=np.zeros(m.N)
        if not avalanche:
            Znew,stats=self.newton(Z,b,None,verbose=verbose)
            return Znew,G,{'outer_iter':1,'outer_converged':stats['newton_converged'],**stats}
        out=[]
        success=False
        for it in range(outerit):
            Znew,stats=self.newton(Z,b,G,verbose=False,tol=3e-5)
            if not stats['newton_converged']:
                break
            psi=Znew[:m.N]
            n=np.exp(np.clip(Znew[m.N:2*m.N],-150,80))
            p=np.exp(np.clip(Znew[2*m.N:],-150,80))
            Gnext,emax,E=m.flux_and_generation(psi,n,p)
            rel_G=np.max(np.abs(Gnext-G))/max(np.max(Gnext),1e10)
            Z=Znew
            if verbose: print('outer II',it,'Gmax',Gnext.max(),'relG',rel_G,'Emax',emax,flush=True)
            out.append((stats['residual'],rel_G,float(Gnext.max()),float(emax)))
            if rel_G<tol:
                G=Gnext;success=True;break
            # choose damping in strong feedback regime
            G=.5*G+.5*Gnext
        return Z,G,{'outer_iter':len(out),'outer_converged':success,'newton':stats,'history':out}

if __name__=='__main__':
    import time,sys
    t=time.time()
    m=AvalancheJFET(a=.5,NA=1e17,dx=.1,dy=.1,Ld=10)
    sol=CoupledNewton(m)
    Z=sol.state0()
    for vg in [-1.,-2.,-3.,-4.]:
        Z,G,st=sol.run(0,vg,Z,avalanche=False)
        print('G',vg,st['newton_converged'],st['iters'],st['residual'],'t',time.time()-t,flush=True)
    for vd in [1,2,5,10,15,20,30,40,50,60,80,100]:
        Z,G,st=sol.run(vd,-4,Z,avalanche=False)
        print('D',vd,st['newton_converged'],st['iters'],st['residual'],'t',time.time()-t,flush=True)
        if not st['newton_converged']: break
        gain,EM=m.feedback_gain(psi=Z[:m.N],steps=20)
        print('GAIN',gain,'EF',EM,flush=True)

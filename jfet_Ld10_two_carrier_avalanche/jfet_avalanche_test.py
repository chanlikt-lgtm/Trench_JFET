import os,sys,time,numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from jfet_newton_dd_robust import AvalancheJFET,CoupledNewton,N0

def precondition(m,s,Z,G):
    # solve both continuity equations with fixed field and lagged G
    psi=Z[:m.N]
    src=s.gamma*G
    mn,rhsn,_,_=m.build_transport(psi,True)
    mp,rhsp,_,_=m.build_transport(psi,False)
    scaleN=sp.diags(1/np.maximum(mn.diagonal(),1e-100))
    scaleP=sp.diags(1/np.maximum(mp.diagonal(),1e-100))
    nn=sla.spsolve(scaleN@mn,scaleN@(rhsn+src))
    pp=sla.spsolve(scaleP@mp,scaleP@(rhsp+src))
    if nn.min()< -1e-4 or pp.min() < -1e-4: print('negative significant during precondition',nn.min(),pp.min(),flush=True)
    nn=np.maximum(nn,np.exp(-40)); pp=np.maximum(pp,np.exp(-40))
    return np.concatenate([psi,np.log(nn),np.log(pp)])

def avalanche_solve(m,s,Zinit,Ginit=None,maxit=80, tol=0.005,damping=.5,printing=False):
    G=np.zeros(m.N) if Ginit is None else Ginit.copy()
    Z=Zinit.copy()
    b=m.contacts_rhs(0,m.Vd,m.Vg)
    results=[]
    for it in range(maxit):
        # fixed-psi transport solve to provide positive log-carrier initializer
        Zp=precondition(m,s,Z,G)
        Z2,status=s.newton(Zp,b,G,maxit=65,tol=2e-5,verbose=False)
        if not status['newton_converged']:
            if printing:print('Newton fail',it,status['residual'],flush=True)
            return Z,G,{'converged':False,'iters':it+1,'reason':'Newton failure','history':results}
        psi=Z2[:m.N];nn=np.exp(Z2[m.N:2*m.N]);pp=np.exp(Z2[2*m.N:])
        Gcalc,Emax,E=m.flux_and_generation(psi,nn,pp)
        if not np.all(np.isfinite(Gcalc)):
            return Z,G,{'converged':False,'iters':it+1,'reason':'Nonfinite G','history':results}
        rel=np.max(np.abs(G-Gcalc))/max(Gcalc.max(),1e10)
        # convergence of ψ and carrier values as well
        dz=np.max(np.abs(Z2[:m.N]-Z[:m.N]))
        if printing and (it%2==0 or rel<tol): print('outer it',it,'relG',rel,'dpsi',dz,'Gmax',Gcalc.max(),'Emax',Emax,flush=True)
        results.append((float(rel),float(dz),float(Gcalc.max()),float(Emax),float(status['residual'])))
        Z=Z2
        if rel<tol and dz<5e-3:
            return Z,Gcalc,{'converged':True,'iters':it+1,'history':results}
        G=(1-damping)*G+damping*Gcalc
    return Z,G,{'converged':False,'iters':maxit,'reason':'outer nonconverged','history':results}

if __name__=='__main__':
    t=time.time()
    m=AvalancheJFET(dx=.1,dy=.1)
    s=CoupledNewton(m);Z=s.state0()
    for vg in np.arange(-.2,-4.01,-.2):Z,G,st=s.run(0,vg,Z,avalanche=False)
    for vd in [1,5,10,15,20,25,30,35,40,42,44,46]:
        Z,_,st=s.run(vd,-4,Z,avalanche=False)
        print('base',vd,st['newton_converged'],st['iters'],st['residual'],'T',round(time.time()-t,1),flush=True)
        if not st['newton_converged']:break
        gain,em=m.feedback_gain(psi=Z[:m.N],steps=20)
        print('gain',gain,'Emax',em,flush=True)
        if vd in [15,20,25,30,35,40,42,44]:
            Z2,G2,st2=avalanche_solve(m,s,Z,maxit=70,damping=.4,printing=False)
            print('avalanche',vd,'conv',st2['converged'],'it',st2['iters'],'last',st2['history'][-1] if st2['history'] else None,'T',round(time.time()-t,1),flush=True)
            if st2['converged']:
                np.savez_compressed(os.path.join(HERE,f'jfet_selfcons_avalanche_{vd}V.npz'),x=m.x,y=m.y,psi=Z2[:m.N].reshape(m.Ny,m.Nx),n=np.exp(Z2[m.N:2*m.N]).reshape(m.Ny,m.Nx)*N0,p=np.exp(Z2[2*m.N:]).reshape(m.Ny,m.Nx)*N0,G=G2.reshape(m.Ny,m.Nx))

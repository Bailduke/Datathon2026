import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt import q
import pandas as pd, numpy as np
rng=np.random.default_rng(0)
cat=pd.DataFrame(q("SELECT product_id::String AS p, subcategory::String AS sub, price_eur::Float64 AS price FROM shop.catalog")["data"])
# posició de preu dins la subcategoria (0 = la més cara, 1 = la més barata), rang mitjà
cat["upos"]=cat.groupby("sub").price.rank(ascending=False, method="average")
cat["upos"]=(cat.upos-0.5)/cat.groupby("sub").p.transform("size")
b=pd.DataFrame(q("""SELECT user_id::String AS u, product_id::String AS p, device::String AS dev, region::String AS reg,
   dateDiff('day', toDateTime64('2026-09-15 07:00:00',3), ts::DateTime64(3)) AS day, toUnixTimestamp64Milli(ts::DateTime64(3)) AS t
   FROM shop.history_events WHERE event_type::String='purchase'""",600)["data"]).merge(cat,on="p").sort_values(["u","t"])
A=b[b.day<30]; B=b[(b.day>=30)&(b.day<45)]; Cc=b[b.day>=45]
pop=A.groupby("p").size(); cat["lpop"]=np.log1p(cat.p.map(pop).fillna(0))
# perfil de preu de l'usuari (període A): residu mitjà u - E[u | popularitat], encongit cap a 0
w=np.exp(cat.lpop); cat["w"]=w
eu=cat.groupby("sub").apply(lambda g: (g.w*g.upos).sum()/g.w.sum(), include_groups=False)
A=A.assign(res=A.upos-A["sub"].map(eu))
g=A.groupby("u").res.agg(["sum","size"]); r_u=(g["sum"]/(g["size"]+3)).to_dict()
bought=A.groupby("u").p.apply(set).to_dict()
opts={s:gg.reset_index(drop=True) for s,gg in cat.groupby("sub")}
devs=sorted(b.dev.unique()); regs=sorted(b.reg.unique())
def feats(rows, max_n):
    X=[];Y=[];M=[]
    for r in rows.itertuples():
        o=opts[r.sub]; bb=bought.get(r.u,set())
        o=o[~o.p.isin(bb)] if bb else o
        if r.p not in set(o.p) or len(o)<3: continue
        k=len(o); up=o.upos.values-0.5; ru=r_u.get(r.u,0.0); known=r.u in r_u
        f=[o.lpop.values, up, up*ru, up*(not known)]
        f+=[up*(r.dev==d) for d in devs[1:]]+[up*(r.reg==x) for x in regs[1:]]
        F=np.zeros((max_n,len(f)),np.float32); F[:k]=np.stack(f,1)
        y=np.zeros(max_n,np.float32); y[np.flatnonzero(o.p.values==r.p)[0]]=1
        m=np.zeros(max_n,bool); m[:k]=True
        X.append(F);Y.append(y);M.append(m)
    return np.stack(X),np.stack(Y),np.stack(M)
MAXN=int(cat.groupby("sub").size().max())
Bs=B.sample(25000,random_state=1); Cs=Cc.sample(25000,random_state=2)
Xb,Yb,Mb=feats(Bs,MAXN); Xc,Yc,Mc=feats(Cs,MAXN)
names=["log_popularitat","preu (barat+)","preu x perfil usuari","preu x usuari nou"]+[f"preu x {d}" for d in devs[1:]]+[f"preu x {x}" for x in regs[1:]]
def fit(cols, X, Y, M, iters=25, l2=1e-3):
    Xs=X[:,:,cols]; beta=np.zeros(len(cols))
    for _ in range(iters):
        eta=np.where(M, Xs@beta, -1e9); eta-=eta.max(1,keepdims=True); p=np.exp(eta)*M; p/=p.sum(1,keepdims=True)
        xbar=(p[:,:,None]*Xs).sum(1); g=((Y[:,:,None]*Xs).sum(1)-xbar).sum(0)-l2*beta
        H=np.einsum("nm,nmi,nmj->ij",p,Xs,Xs)-xbar.T@xbar+l2*np.eye(len(cols))
        step=np.linalg.solve(H,g); beta+=step
        if np.abs(step).max()<1e-6: break
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))
def evaluate(cols,beta,X,Y,M):
    eta=np.where(M, X[:,:,cols]@beta, -1e9); eta-=eta.max(1,keepdims=True); p=np.exp(eta)*M; p/=p.sum(1,keepdims=True)
    ll=np.log((p*Y).sum(1)+1e-12).mean(); top=np.argsort(-p,1)[:,:3]; hr=np.take_along_axis(Y,top,1).sum(1).mean()
    return ll,hr
n_opts=Mc.sum(1); print(f"eleccions: ajust {len(Yb)} · avaluació {len(Yc)} · opcions per elecció (mitjana) {n_opts.mean():.0f} · LL a l'atzar {-np.log(n_opts).mean():.3f}")
models={"M0 popularitat":[0],"M1 + preu":[0,1],"M2 + perfil de preu de l'usuari":[0,1,2,3],"M3 + device i regió":list(range(len(names)))}
for nm,cols in models.items():
    beta,se=fit(cols,Xb,Yb,Mb); ll,hr=evaluate(cols,beta,Xc,Yc,Mc)
    print(f"\n{nm:<34} LL={ll:.4f}  HR@3={hr:.3f}")
    for c,bv,s in zip(cols,beta,se):
        if nm.startswith("M3") and c>=4 and abs(bv/s)<2: continue
        print(f"    {names[c]:<28} {bv:+.3f}  (z={bv/s:+.1f})")
    if nm.startswith("M3"): print("    (interaccions de device/regió amb |z|<2 amagades)")

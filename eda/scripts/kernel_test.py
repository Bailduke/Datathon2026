import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rt import q
import pandas as pd, numpy as np
CUT="2026-10-30 07:00:00"
d=pd.DataFrame(q(f"""WITH c AS (SELECT product_id::String AS pid, subcategory::String AS sub, category::String AS cat FROM shop.catalog)
 SELECT e.u AS u, c.sub AS sub, any(c.cat) AS cat,
   sumIf(if(e.t='purchase',1,if(e.t='add_to_cart',0.25,0.02)), e.past=1) AS w_past,
   countIf(e.t='purchase' AND e.past=1) AS p_past,
   countIf(e.t='purchase' AND e.past=0) AS p_fut
 FROM (SELECT user_id::String AS u, product_id::String AS pid, event_type::String AS t,
         toUInt8(ts::DateTime64(3) < toDateTime64('{CUT}',3)) AS past FROM shop.history_events WHERE cityHash64(user_id::String) % 2 = 0) AS e
 INNER JOIN c ON c.pid = e.pid GROUP BY u, sub""",900)["data"])
subs=sorted(d["sub"].unique()); s_idx={s:i for i,s in enumerate(subs)}
sub_cat=d.drop_duplicates("sub").set_index("sub").cat; cats=sorted(sub_cat.unique()); c_idx={c:i for i,c in enumerate(cats)}
sc=np.array([c_idx[sub_cat[s]] for s in subs])
users=d.u.unique(); u_idx={u:i for i,u in enumerate(users)}
NU,NS,NC=len(users),len(subs),len(cats)
W=np.zeros((NU,NS)); Pp=np.zeros((NU,NS)); F=np.zeros((NU,NS))
ui=d.u.map(u_idx).values; si=d["sub"].map(s_idx).values
W[ui,si]=d.w_past.values; Pp[ui,si]=d.p_past.values; F[ui,si]=d.p_fut.values
# nuclis amb dades del PASSAT (sense fuites): correlació phi de compra sí/no entre usuaris
def kernel(B):
    C=np.corrcoef(B.T); np.fill_diagonal(C,0); C=np.nan_to_num(np.clip(C,0,None))
    return C/np.maximum(C.sum(1,keepdims=True),1e-12)
Bs=(Pp>0).astype(float)
Bc=np.stack([Bs[:,sc==k].max(1) for k in range(NC)],1)
Kc=kernel(Bc)                                   # 10x10, model "categoria general"
Ks=np.zeros((NS,NS))                            # 10 models dins de cada categoria (bloc diagonal)
for k in range(NC):
    ix=np.flatnonzero(sc==k); Ks[np.ix_(ix,ix)]=kernel(Bs[:,ix])
Wc=np.stack([W[:,sc==k].sum(1) for k in range(NC)],1)
m_c=Wc.sum(0)/Wc.sum(); m_s_in=np.array([W[:,j].sum()/W[:,sc==sc[j]].sum() for j in range(NS)])
def predict(lam, kappa_c=2.0, kappa_s=2.0):
    Ac=Wc + lam*(Wc@Kc)                         # suma la fila de correlació de cada categoria triada
    Pc=(Ac + kappa_c*m_c)/(Ac.sum(1,keepdims=True)+kappa_c)
    As=W + lam*(W@Ks)
    Ps=np.zeros_like(W)
    for k in range(NC):
        ix=np.flatnonzero(sc==k); a=As[:,ix]
        Ps[:,ix]=(a + kappa_s*m_s_in[ix])/(a.sum(1,keepdims=True)+kappa_s)
    return Ps*Pc[:,sc]                          # P(sub|u) = P(cat|u) P(sub|cat,u)
has_f=F.sum(1)>0; act=W.sum(1)
groups={"poca historia (<=1)":(act<=1),"mitjana (1-10]":(act>1)&(act<=10),"molta (>10)":act>10,"tots":act>=0}
print(f"usuaris amb compres futures: {has_f.sum()} · subcategories {NS} · categories {NC}")
print(f"{'lambda':>7} | " + " | ".join(f"{g:>22}" for g in groups))
for lam in [0,0.25,0.5,1,2,4]:
    P=predict(lam); out=[]
    for g,mask in groups.items():
        m=mask&has_f
        ll=(F[m]*np.log(P[m]+1e-12)).sum()/F[m].sum()
        top=np.argsort(-P[m],1)[:,:3]; hit=np.take_along_axis(F[m],top,1).sum()/F[m].sum()
        out.append(f"LL {ll:6.3f} HR@3 {hit:.3f}")
    print(f"{lam:>7} | " + " | ".join(f"{o:>22}" for o in out))

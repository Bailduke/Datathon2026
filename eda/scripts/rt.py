import os, sys, json, requests
from pathlib import Path
for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
    k,_,v=line.partition("="); 
    if k.strip(): os.environ.setdefault(k.strip(), v.strip())
H={"Authorization": f"Bearer {os.environ['RAWTREE_API_KEY']}"}
TEAM=os.environ["TEAM_ID"]
def q(sql, timeout=300):
    r=requests.post("https://api.rawtree.com/v1/query", json={"sql":sql}, headers=H, timeout=timeout)
    if r.status_code!=200: return {"error": r.text[:500]}
    return r.json()
if __name__=="__main__":
    for s in sys.argv[1:]:
        res=q(s.replace("team_NN",TEAM))
        print("##", s); print(json.dumps(res.get("data",res), default=str, ensure_ascii=False)[:4000])

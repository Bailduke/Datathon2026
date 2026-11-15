"""Guarda una foto del leaderboard público cada 5 minutos en eda/leaderboard_log.jsonl (una línea por foto)."""
import json, time, urllib.request, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "leaderboard_log.jsonl")
URL = "https://datathon.rawtree.dev/api/state"
def snap():
    d = json.load(urllib.request.urlopen(URL, timeout=20)); m = json.load(urllib.request.urlopen(URL + "?by=model", timeout=20))
    keep = lambda x, ks: {k: x.get(k) for k in ks}
    return {"ts": time.time(), "window": d.get("window"), "mode": d.get("mode"),
            "baseline": keep(d["baseline"], ("conversion", "cold_conversion", "requests")),
            "reference": keep(d["reference"], ("conversion", "cold_conversion", "requests")),
            "teams": [keep(t, ("team_id", "requests", "conversion", "cold_conversion", "on_time", "model_version")) for t in d["teams"]],
            "versions": [dict(keep(v, ("model_version", "requests", "cold_requests", "conversion", "cold_conversion", "first", "last")), team=t["team_id"])
                         for t in m["teams"] for v in (t.get("versions") or [])]}
while True:
    try:
        with open(OUT, "a") as f: f.write(json.dumps(snap()) + "\n")
    except Exception as e:
        with open(OUT + ".err", "a") as f: f.write(f"{time.strftime('%H:%M:%S')} {str(e)[:200]}\n")
    time.sleep(300)

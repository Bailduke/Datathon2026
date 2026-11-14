"""Etiquetas (atributos) escondidas en la descripción de cada producto."""
import re, collections, pickle, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TAGS = ["compact and lightweight", "a family-size format", "a luxury feel", "a sporty style", "premium materials",
        "a minimalist design", "smart app control", "bold colours", "organic ingredients", "made from recycled materials",
        "designed for kids", "wireless connectivity", "great value for money", "built for professionals",
        "whisper-quiet operation", "a handmade finish", "a vintage look", "a waterproof shell", "extra durability",
        "a travel-ready build"]
def parse(desc):
    d = (desc or "").lower()
    return [k for k, t in enumerate(TAGS) if t in d]
if __name__ == "__main__":
    from rt import q
    allr = q("SELECT product_id::String AS p, toString(description) AS d, toString(brand) AS b, toString(subcategory) AS s FROM shop.catalog", 120)["data"]
    n = collections.Counter(); rest = collections.Counter(); tagc = collections.Counter()
    for r in allr:
        ts = parse(r["d"]); n[len(ts)] += 1
        for t in ts: tagc[t] += 1
        if len(ts) != 2:
            d = r["d"].lower()
            for t in TAGS: d = d.replace(t, "#")
            d = d.replace(r["b"].lower(), "B").replace(r["s"].lower(), "S"); rest[d] += 1
    print("nº de etiquetas por producto:", dict(n)); print("frecuencia por etiqueta:", sorted(tagc.values()))
    for d, c in rest.most_common(12): print("  resto:", c, d)
    tpl = collections.Counter()
    for r in allr:
        d = r["d"].lower()
        for t in TAGS: d = d.replace(t, "#")
        tpl[d.replace(r["b"].lower(), "B").replace(r["s"].lower(), "S")] += 1
    print("plantillas:"); [print("  ", c, t) for t, c in tpl.most_common(12)]
    cu = q("SELECT * FROM shop.catalog_updates WHERE change::String = 'new_product' LIMIT 2", 60)["data"]
    print("alta de producto nuevo:", cu[:2])
    pickle.dump({r["p"]: parse(r["d"]) for r in allr}, open(os.path.join(HERE, ".cache", "tags.pkl"), "wb"))

import json

path = r"Codes/Host Classification/Host_Classification_Final_merged.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

# Print only CODE cells (skip markdown) to keep the dump compact
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    print("--- CELL %d [code] ---" % i)
    print(src)
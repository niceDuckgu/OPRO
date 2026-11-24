"""Print the validation accuracy of every run found under an output directory."""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
rows = []
for path in sorted(root.glob("*/result.json")):
    result = json.loads(path.read_text())
    params = result["params"]
    rows.append((path.parent.name, result["final_val_acc"], result["best_val_acc"], params["lora"], params["opro"]))

if not rows:
    sys.exit(f"no result.json under {root}/")
width = max(len(r[0]) for r in rows)
print(f"{'run':<{width}}  final  best   lora params  opro params")
for name, final, best, lora, opro in rows:
    print(f"{name:<{width}}  {final:5.2f}  {best:5.2f}  {lora:>11,}  {opro:>11,}")

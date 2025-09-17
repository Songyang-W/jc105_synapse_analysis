import sys
import math
import re
import subprocess
from pathlib import Path

ID_FILE = Path("root_ids_version24.txt")

# --- Load & normalize IDs (works for comma-separated OR one-per-line) ---
text = ID_FILE.read_text(encoding="utf-8")
# split on commas or any whitespace, drop empties
ids = [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]

n_ids = len(ids)
n_groups = 8
group_size = math.ceil(n_ids / n_groups)

# get task index from qsub (1-based)
task_id = int(sys.argv[1])  # e.g. from $SGE_TASK_ID
start = (task_id - 1) * group_size
end = min(start + group_size, n_ids)

my_ids = ids[start:end]

print(f"Task {task_id}: processing {len(my_ids)} IDs")
print(my_ids)

# call your real script, passing IDs
# you might adapt this depending on how manual_split_axon_dendrite_loop.py expects input
import subprocess
subprocess.run(["python", "manual_split_axon_dendrite_loop.py", *my_ids])

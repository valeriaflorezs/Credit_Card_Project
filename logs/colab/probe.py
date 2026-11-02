import os
import subprocess
import sys

print("nproc:", os.cpu_count())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
print("python:", sys.version)
print(subprocess.run(["bash", "-c", "free -g | head -2"], capture_output=True, text=True).stdout)

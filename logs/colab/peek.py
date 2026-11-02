import glob, os, subprocess
for f in glob.glob("/content/credit_card/run_*.log"):
    print("==", f)
    print(subprocess.run(["bash", "-c", f"grep -a experiment_runner {f} | tail -6"], capture_output=True, text=True).stdout)
for f in glob.glob("/content/credit_card/results/experiments_5x5__colab_*.csv"):
    n = sum(1 for _ in open(f)) - 1
    print(f, "filas:", n)
print(subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime"], capture_output=True, text=True).stdout)

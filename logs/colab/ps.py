import os, subprocess
def sh(c): 
    r = subprocess.run(["bash", "-c", c], capture_output=True, text=True); return (r.stdout + r.stderr)[-1800:]
print(sh("ps -eo pid,etimes,pcpu,args | grep run_colab_5x5 | grep -v grep"))
for f in ("/content/cuml_failed", "/content/cuml_ready"):
    if os.path.exists(f): print(f, open(f).read()[-1200:])
print(sh("ls -la /content/credit_card/results 2>/dev/null"))

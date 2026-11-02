"""Se ejecuta DENTRO de la VM: instala cuML (solo cu12), arregla la ruta de carga y
deja /content/cuml_ready (o /content/cuml_failed). Corre en segundo plano."""
import subprocess
import sys

SCRIPT = r'''
import glob, os, subprocess, sys
py = sys.executable
def sh(c, env=None):
    return subprocess.run(["bash", "-c", c], capture_output=True, text=True, env=env)
sh(f"{py} -m pip uninstall -y cuml-cu12 libcuml-cu12 cudf-cu12 cupy-cuda12x rmm-cu12 libraft-cu12 pylibraft-cu12 libcudf-cu12 librmm-cu12 pylibcudf-cu12")
r = sh(f"{py} -m pip install --extra-index-url=https://pypi.nvidia.com cuml-cu13==26.6.* libcuml-cu13==26.6.* libnvforest-cu13==26.6.* libraft-cu13==26.6.* pylibraft-cu13==26.6.* rmm-cu13==26.6.* librmm-cu13==26.6.* cudf-cu13==26.6.* libcudf-cu13==26.6.* pylibcudf-cu13==26.6.* rapids-logger \"nvidia-cuda-runtime<13.1\"")
sp = [p for p in glob.glob("/usr/local/lib/python3.*/dist-packages")][0]
dirs = [d for d in glob.glob(sp + "/*/lib64") + glob.glob(sp + "/*/lib") + glob.glob(sp + "/nvidia/*/lib") if "cu12" not in d]
ld = ":".join(dirs)
env = dict(os.environ, LD_LIBRARY_PATH=ld)
code = "import cuml, numpy as np; from cuml.svm import SVC; X=np.random.rand(500,5).astype('float32'); y=(X[:,0]>.5).astype(int); m=SVC(kernel='rbf',output_type='numpy').fit(X,y); print('CUML_OK', m.decision_function(X).shape)"
t = sh(f"{py} -c \"{code}\" 2>&1 | tail -3", env)
open("/content/ld_path.txt", "w").write(ld)
open("/content/cuml_ready" if "CUML_OK" in t.stdout else "/content/cuml_failed", "w").write(t.stdout + t.stderr + r.stdout[-500:] + r.stderr[-500:])
'''
open("/content/cuml13_setup_run.py", "w").write(SCRIPT)
subprocess.Popen([sys.executable, "/content/cuml13_setup_run.py"], start_new_session=True)
print("cuML se instala en segundo plano")

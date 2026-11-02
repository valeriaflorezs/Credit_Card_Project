"""DENTRO de la VM: instala RAPIDS con el instalador oficial de Colab (el mismo del notebook 07
que funciono) y deja /content/cuml_ready o /content/cuml_failed."""
import subprocess, sys
SCRIPT = r'''
import os, subprocess, sys
def sh(c):
    return subprocess.run(["bash", "-c", c], capture_output=True, text=True)
for f in ("/content/cuml_ready", "/content/cuml_failed"):
    os.path.exists(f) and os.remove(f)
a = sh("cd /content && rm -rf rapidsai-csp-utils && git clone --depth 1 https://github.com/rapidsai/rapidsai-csp-utils.git && python rapidsai-csp-utils/colab/pip-install.py")
open("/content/rapids_install.log", "w").write(a.stdout[-4000:] + a.stderr[-2000:])
b = sh("pip install -q optuna deap")
code = "import cuml; from cuml.svm import SVC; import numpy as np; X=np.random.rand(500,5).astype('float32'); y=(X[:,0]>.5).astype(int); m=SVC(kernel='rbf',probability=True,output_type='numpy').fit(X,y); print('CUML_OK', cuml.__version__, m.predict_proba(X).shape)"
t = sh(f"{sys.executable} -c \"{code}\" 2>&1 | tail -4")
open("/content/cuml_ready" if "CUML_OK" in t.stdout else "/content/cuml_failed", "w").write(t.stdout + t.stderr)
'''
open("/content/rapids_setup_run.py", "w").write(SCRIPT)
subprocess.Popen([sys.executable, "/content/rapids_setup_run.py"], start_new_session=True)
print("RAPIDS se instala en segundo plano")

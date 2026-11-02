import glob, os, subprocess, sys
def sh(c, env=None):
    r = subprocess.run(["bash", "-c", c], capture_output=True, text=True, env=env); return (r.stdout + r.stderr)[-1200:]
print(sh("pip install -q --extra-index-url=https://pypi.nvidia.com libnvforest-cu12 rapids-logger 2>&1 | tail -3"))
sp = "/usr/local/lib/python3.13/dist-packages"
dirs = [d for d in glob.glob(sp + "/*/lib64") + glob.glob(sp + "/*/lib") + glob.glob(sp + "/nvidia/*/lib") if "cu13" not in d]
env = dict(os.environ, LD_LIBRARY_PATH=":".join(dirs))
print(sh("f=/usr/local/lib/python3.13/dist-packages/cuml/internals/logger.abi3.so; ldd $f | grep -i 'not found'", env))
code = "import cuml, numpy as np; from cuml.svm import SVC; X=np.random.rand(500,5).astype('float32'); y=(X[:,0]>.5).astype(int); m=SVC(kernel='rbf',probability=True,output_type='numpy').fit(X,y); print('CUML_OK', cuml.__version__, m.predict_proba(X).shape)"
print(sh(f"{sys.executable} -c \"{code}\" 2>&1 | tail -4", env))
open("/content/ld_path.txt", "w").write(env["LD_LIBRARY_PATH"])

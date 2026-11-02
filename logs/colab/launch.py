"""Se ejecuta DENTRO de la VM de Colab (colab exec -f): descomprime el paquete y
lanza la corrida en segundo plano (sobrevive a que termine este exec)."""
import os
import subprocess
import sys
import zipfile

MODEL_ARGS = os.environ.get("MODEL_ARGS", "xgboost")
DEVICE = os.environ.get("XGB_DEVICE", "cuda")

if subprocess.run(["bash", "-c", f"pgrep -f 'src.run_colab_5x5 {MODEL_ARGS}'"], capture_output=True).returncode == 0:
    print("ya hay una corrida de", MODEL_ARGS, "; no se lanza otra")
    raise SystemExit
work = "/content/credit_card"
if not os.path.exists(work + "/src"):
    zipfile.ZipFile("/content/colab_package.zip").extractall(work)
os.makedirs(work + "/results", exist_ok=True)
if os.path.exists("/content/resume.csv"):  # reanudar lo ya descargado tras recrear la VM
    import shutil
    shutil.copy("/content/resume.csv", work + "/results/experiments_5x5__colab_" + MODEL_ARGS.split()[0] + ".csv")
log = open(work + f"/run_{MODEL_ARGS.replace(' ', '-')}.log", "ab")
env = dict(os.environ, XGB_DEVICE=DEVICE, PYTHONUNBUFFERED="1")
if os.environ.get("SVM_BACKEND") == "cuml":
    env["SVM_BACKEND"] = "cuml"
    env["N_JOBS"] = "1"  # la GPU hace el trabajo pesado; hilos concurrentes sobre cuML no son seguros
if "knn" in MODEL_ARGS.split():
    env["KNN_BACKEND"] = "cuml"
    env["N_JOBS"] = "1"
p = subprocess.Popen(
    [sys.executable, "-u", "-m", "src.run_colab_5x5", *MODEL_ARGS.split()],
    cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
print("lanzado pid", p.pid, "modelos", MODEL_ARGS, "device", DEVICE)

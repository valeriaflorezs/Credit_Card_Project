"""Respaldo: cuando termina XGBoost local, si KNN de Colab no esta completo (<64 combos, 4 tecnicas x 4 optimizadores x 4... se mide por filas OK),
corre KNN local con reanudacion. Se ejecuta en segundo plano."""
import subprocess, time, pathlib, csv

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_COLAB = ROOT / "results" / "experiments_5x5__colab_knn.csv"
CSV_LOCAL = ROOT / "results" / "experiments_5x5__local.csv"


def running(pattern):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? {{ $_.CommandLine -like '*{pattern}*' }} | Measure-Object).Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def n_rows(path, model):
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f) if r.get("model") == model)


# esperar a que XGBoost local haya empezado y terminado
while not running("run_local_5x5 xgboost"):
    time.sleep(120)
while running("run_local_5x5 xgboost"):
    time.sleep(120)

if n_rows(CSV_COLAB, "knn") < 16 and n_rows(CSV_LOCAL, "knn") < 16:
    with open(ROOT / "logs" / "run_local_5x5_knn.log", "a") as log:
        subprocess.run(["C:/Users/valef/anaconda3/envs/creditcard/python.exe", "-u", "-m", "src.run_local_5x5", "knn"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


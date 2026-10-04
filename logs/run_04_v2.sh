#!/bin/bash
# Corrida v2 (dataset preprocesado, sin SVM): reintenta si el proceso muere; resume=True salta lo ya guardado.
cd "$(dirname "$0")/../notebooks" || exit 1
PY="/c/Users/valef/anaconda3/envs/creditcard/python.exe"
export PYTHONIOENCODING=utf-8
for i in 1 2 3 4 5 6; do
  n=$("$PY" -c "import os,pandas as pd; p='../results/experiments_master.csv'; print(len(pd.read_csv(p)) if os.path.exists(p) else 0)")
  echo "$(date '+%F %T') intento $i: $n/96 filas guardadas"
  [ "$n" -ge 96 ] && echo "COMPLETO" && break
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 04_run_experiments.ipynb > ../logs/04_v2_intento$i.log 2>&1
  echo "$(date '+%F %T') intento $i terminó con código $?"
done
n=$("$PY" -c "import os,pandas as pd; p='../results/experiments_master.csv'; print(len(pd.read_csv(p)) if os.path.exists(p) else 0)")
echo "$(date '+%F %T') FIN: $n/96 filas"

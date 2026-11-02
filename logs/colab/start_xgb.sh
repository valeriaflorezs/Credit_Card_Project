#!/bin/bash
# Sube el paquete, instala dependencias y lanza XGBoost (GPU) en la sesion "xgb".
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
S=${1:-xgb}
$C upload -s $S $P/colab_package.zip /content/colab_package.zip 2>&1 | tail -2
$C install -s $S -r $P/logs/colab/req_min.txt 2>&1 | tail -6
{ echo "import os; os.environ['MODEL_ARGS']='${2:-xgboost}'; os.environ['XGB_DEVICE']='${3:-cuda}'; os.environ['SVM_BACKEND']='${4:-}'"; cat $P/logs/colab/launch.py; } > /tmp/launch_$S.py
$C exec -s $S -f /tmp/launch_$S.py 2>&1 | tail -4

#!/bin/bash
# Uso: relaunch.sh svm|knn  -> solo relanza la corrida en una sesion que ya tiene RAPIDS y el paquete.
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
S=$1
{ echo "import os; os.environ['MODEL_ARGS']='$S'; os.environ['XGB_DEVICE']='cpu'; os.environ['SVM_BACKEND']='cuml'"; cat $P/logs/colab/launch.py; } > /tmp/launch_$S.py
timeout 170 $C exec -s $S -f /tmp/launch_$S.py 2>&1 | tail -3

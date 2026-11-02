#!/bin/bash
# Uso: setup_session.sh svm|knn  -> crea la T4, instala RAPIDS (instalador oficial), verifica y lanza la corrida.
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
S=$1
LOCK=$P/logs/colab/setup_$S.lock
[ -e $LOCK ] && exit 0
touch $LOCK; trap "rm -f $LOCK" EXIT
OUTN=$(timeout 300 $C new -s $S --gpu T4 2>&1 | tail -2); echo "$OUTN"
echo "$OUTN" | grep -q -E 'READY|already' || { echo 'no se pudo crear la sesion'; exit 1; }
timeout 120 $C exec -s $S -f $P/logs/colab/rapids_setup.py 2>&1 | tail -1
for i in $(seq 1 90); do
  sleep 20
  M=$(timeout 60 $C exec -s $S -f $P/logs/colab/marker.py 2>&1 | grep -o 'MARKER [a-z]*')
  echo "$M"
  [ "$M" = "MARKER ready" ] && break
  [ "$M" = "MARKER failed" ] && { echo "cuML fallo"; exit 1; }
done
$C upload -s $S $P/colab_package.zip /content/colab_package.zip 2>&1 | tail -1
[ -f $P/results/experiments_5x5__colab_$S.csv ] && $C upload -s $S $P/results/experiments_5x5__colab_$S.csv /content/resume.csv 2>&1 | tail -1
# SIN instalar nada mas con pip: cambiar numpy rompe RAPIDS.
{ echo "import os; os.environ['MODEL_ARGS']='$S'; os.environ['XGB_DEVICE']='cpu'; os.environ['SVM_BACKEND']='cuml'"; cat $P/logs/colab/launch.py; } > /tmp/launch_$S.py
$C exec -s $S -f /tmp/launch_$S.py 2>&1 | tail -3

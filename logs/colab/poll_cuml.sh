#!/bin/bash
C=/root/colab/bin/colab
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
rm_marker="import os
for f in ('/content/cuml_ready','/content/cuml_failed'):
    os.path.exists(f) and os.remove(f)"
printf "$rm_marker
" | timeout 60 $C exec -s svm 2>&1 | tail -1
timeout 120 $C exec -s svm -f $P/logs/colab/$1 2>&1 | tail -1
for i in $(seq 1 45); do
  sleep 20
  M=$(timeout 60 $C exec -s svm -f $P/logs/colab/marker.py 2>&1 | grep -o 'MARKER [a-z]*')
  echo "$M"
  [ "$M" = "MARKER ready" ] && break
  [ "$M" = "MARKER failed" ] && break
done
timeout 60 $C exec -s svm -f $P/logs/colab/ps.py 2>&1 | grep -a -A8 cuml_ | head -14

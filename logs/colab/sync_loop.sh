#!/bin/bash
# Cuidador: cada 60 s hace latido (exec minimo) a cada sesion de Colab, baja los CSV a results/ cada ~2 min,
# actualiza logs/estado_colab.txt y, si una sesion desaparece, la recrea y relanza la corrida (reanuda sola
# desde lo que ya se haya descargado, subiendo ese CSV a la nueva VM).
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
SESSIONS="svm"
n=0
while true; do
  n=$((n+1))
  OUT=$P/logs/estado_colab.txt.tmp
  echo "ciclo $n $(date +%H:%M:%S)" > $OUT
  for S in $SESSIONS; do
    echo "### sesion $S" >> $OUT
    if [ -e $P/logs/colab/setup_$S.lock ]; then echo "en preparacion (setup en curso)" >> $OUT; continue; fi
    R=$(timeout 120 $C exec -s $S -f $P/logs/colab/peek.py 2>&1); RC=$?
    echo "$R" >> $OUT
    # Solo se considera perdida si el servidor lo dice; un timeout o un error de red NO recrea la sesion.
    if ! echo "$R" | grep -q -E "appears to be lost|not found"; then
      if [ $((n % 2)) -eq 0 ]; then
        for f in $(timeout 60 $C ls -s $S /content/credit_card/results 2>/dev/null | grep -o 'experiments_5x5__colab_[A-Za-z0-9_.-]*\.csv'); do
          timeout 120 $C download -s $S /content/credit_card/results/$f $P/results/$f >/dev/null 2>&1
        done
      fi
    else
      echo "SESION $S PERDIDA $(date +%H:%M:%S): se recrea" >> $OUT
      echo "SESION $S PERDIDA $(date)" >> $P/logs/colab/incidentes.log
      ( bash $P/logs/colab/setup_session.sh $S >> $P/logs/colab/setup_$S.log 2>&1 ) &
      sleep 5
    fi
  done
  timeout 60 $C usage >> $OUT 2>&1
  mv $OUT $P/logs/estado_colab.txt
  sleep 60
done

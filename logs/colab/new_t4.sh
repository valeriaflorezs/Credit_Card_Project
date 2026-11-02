#!/bin/bash
# Crea la sesion T4 "xgb" y muestra hardware y tasa de consumo.
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
$C new -s xgb --gpu T4 2>&1 | tail -5
$C status -s xgb 2>&1 | tail -8
$C exec -s xgb -f $P/logs/colab/probe.py 2>&1 | tail -12
$C usage 2>&1 | tail -3

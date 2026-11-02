#!/bin/bash
# Instala cuML en la sesion $1 y prueba que importe.
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
$C exec -s $1 -f $P/logs/colab/pip_cuml.py 2>&1 | tail -15

#!/bin/bash
# Uso: new_session.sh NOMBRE GPU   -> crea sesion y muestra hardware/consumo
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
$C new -s $1 --gpu ${2:-T4} 2>&1 | tail -5
$C exec -s $1 -f $P/logs/colab/probe.py 2>&1 | tail -10
$C usage 2>&1 | tail -3

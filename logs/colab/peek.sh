#!/bin/bash
C=/root/colabenv/bin/colab
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project
$C exec -s ${1:-xgb} -f $P/logs/colab/peek.py 2>&1 | tail -25
$C usage 2>&1 | tail -3

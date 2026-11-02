#!/bin/bash
timeout 120 /root/colabenv/bin/colab exec -s xgb -f /mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project/logs/colab/kill_dup.py 2>&1 | tail -5

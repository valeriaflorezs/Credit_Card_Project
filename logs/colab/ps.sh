#!/bin/bash
timeout 120 /root/colabenv/bin/colab exec -s $1 -f /mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project/logs/colab/ps.py 2>&1 | tail -30

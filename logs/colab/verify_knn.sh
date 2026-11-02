#!/bin/bash
timeout 600 /root/colabenv/bin/colab exec -s ${1:-svm} -f /mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project/logs/colab/verify_knn.py 2>&1 | tail -12

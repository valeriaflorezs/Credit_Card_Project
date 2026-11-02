#!/bin/bash
/root/colabenv/bin/colab exec -s ${1:-svm} -f /mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project/logs/colab/cuml_log.py 2>&1 | tail -15

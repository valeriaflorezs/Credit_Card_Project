#!/bin/bash
P=/mnt/c/Users/valef/OneDrive/Desktop/Credit_Card_Project/logs/colab
exec bash $P/setup_session.sh $1 > $P/setup_$1.log 2>&1

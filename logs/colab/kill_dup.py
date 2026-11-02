import subprocess
print(subprocess.run(["bash", "-c", "kill 9596; sleep 2; ps -eo pid,etimes,args | grep run_colab_5x5 | grep -v grep"], capture_output=True, text=True).stdout)

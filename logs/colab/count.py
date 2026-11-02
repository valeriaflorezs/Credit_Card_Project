import subprocess
t = open("/content/credit_card/run_xgboost.log", errors="ignore").read()
print("warnings predict:", t.count("Potential solutions"), "| lineas:", t.count("\n"), "| OK:", t.count("] OK "))

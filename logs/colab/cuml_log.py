t = open("/content/cuml_install.log", errors="ignore").read()
print([l[:120] for l in t.splitlines() if "PIP_RC" in l or "CUML_OK" in l or "Error" in l][-6:])

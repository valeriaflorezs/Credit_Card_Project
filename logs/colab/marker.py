import os

print("MARKER", "ready" if os.path.exists("/content/cuml_ready") else "failed" if os.path.exists("/content/cuml_failed") else "wait")

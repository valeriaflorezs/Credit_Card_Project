import subprocess
import sys

py = sys.executable
cmd = (
    f"{py} -m pip uninstall -y cupy-cuda12x cupy-cuda13x cuml-cu12 cuml-cu13 cudf-cu12 cudf-cu13 libcuml-cu12 libcuml-cu13 "
    f"rmm-cu12 rmm-cu13 raft-dask-cu12 pylibraft-cu12 pylibraft-cu13 libraft-cu12 libraft-cu13 > /content/cuml_install.log 2>&1; "
    f"{py} -m pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 libcuml-cu12 >> /content/cuml_install.log 2>&1; "
    f"echo PIP_RC=$? >> /content/cuml_install.log; "
    f"{py} -c \"import cuml; print('CUML_OK', cuml.__version__)\" >> /content/cuml_install.log 2>&1; "
    f"{py} -m pip list 2>/dev/null | grep -i -E 'cuml|cupy|cudf|rmm|raft' >> /content/cuml_install.log"
)
subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
print("instalando en segundo plano")

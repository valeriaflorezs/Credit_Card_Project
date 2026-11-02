import os, zipfile
os.makedirs("/content/credit_card", exist_ok=True)
zipfile.ZipFile("/content/colab_package.zip").extractall("/content/credit_card")
print("zip extraido")

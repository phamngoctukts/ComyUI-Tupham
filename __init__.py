from .ghepanh import NODE_CLASS_MAPPINGS as CL1, NODE_DISPLAY_NAME_MAPPINGS as NM1
from .utils import store as _
import importlib
import os
import subprocess
import sys

NODE_CLASS_MAPPINGS = {**CL1}
NODE_DISPLAY_NAME_MAPPINGS = {**NM1}
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']

python = sys.executable

def is_installed(package, package_overwrite=None):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        pass

    package = package_overwrite or package

    if spec is None:
        print(f"Installing {package}...")
        command = f'"{python}" -m pip install {package}'
  
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ)

        if result.returncode != 0:
            print(f"Couldn't install\nCommand: {command}\nError code: {result.returncode}")


# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    try:
        if os.name == 'nt':  # Windows
            import shutil
            shutil.copy(src, target)
        else:  # Linux/macOS
            os.system(f'ln -s {src} {target}')
    except Exception as e:
        print(f"[WARNING] link_file failed: {e}")



def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed
    embed()

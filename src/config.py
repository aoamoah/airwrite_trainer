import random
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def preload_cuda_libs():
    """Preload pip-installed NVIDIA libraries so TensorFlow's dlopen-by-soname
    finds them all. TF resolves most of them through wheel RPATHs but misses
    some (observed: libcusolver.so.11), which silently disables the GPU."""
    import ctypes
    import glob

    lib_root = Path(sys.prefix) / "lib"
    pattern = str(lib_root / "python*" / "site-packages" / "nvidia" / "*" / "lib" / "lib*.so*")
    for lib in sorted(glob.glob(pattern)):
        try:
            ctypes.CDLL(lib)
        except OSError:
            pass


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

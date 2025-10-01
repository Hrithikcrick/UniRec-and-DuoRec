import numpy as np

# Restore NumPy aliases removed in 2.x for legacy libs
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

try:
    import torch
except Exception:
    torch = None

if torch is not None:
    _torch_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _load_compat
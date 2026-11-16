"""Strict repeatability settings shared by formal training and frozen inference."""
import os

def configure_torch():
    # Must be set before the first CUDA matrix operation in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1)
    return {"deterministic_algorithms":True,"cublas_workspace_config":os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cudnn_benchmark":False,"tf32":False}

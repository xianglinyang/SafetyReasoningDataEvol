import torch

def str2dtype(dtype_str: str) -> torch.dtype:
    dtype_mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp64": torch.float64,
        "float64": torch.float64,
        "double": torch.float64
    }
    
    dtype_str = dtype_str.lower()
    if dtype_str not in dtype_mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Supported dtypes are: {list(dtype_mapping.keys())}"
        )
    
    return dtype_mapping[dtype_str] 
import psutil
import random

def get_memory_usage() -> float:
    """Returns memory usage percentage."""
    return psutil.virtual_memory().percent

def should_sample() -> tuple[bool, str]:
    """
    Returns (should_keep: bool, state: str)
    < 70% memory -> 100% logging (keep=True, NORMAL)
    70~85% -> 10~30% sampling (keep=probabilistic, SAMPLED)
    > 85% -> CRITICAL / ERROR only (keep=False, CRITICAL_ONLY)
    """
    mem = get_memory_usage()
    
    if mem < 70.0:
        return True, "NORMAL"
    elif mem < 85.0:
        # Linear scale sampling: 70% -> 30% sample rate, 85% -> 10% sample rate
        ratio = 0.3 - ((mem - 70.0) / 15.0) * 0.2
        keep = random.random() < ratio
        return keep, "SAMPLED"
    else:
        return False, "CRITICAL_ONLY"

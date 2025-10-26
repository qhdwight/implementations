import numpy as np


def line_search(x: np.ndarray, dx: np.ndarray) -> tuple[float, int]:
    assert len(x) == len(dx) and len(x) > 0 and len(dx) > 0
    alpha_val = np.inf
    p_index = -1
    for i in range(len(x)):
        if dx[i] <= 0.0:
            continue
        ratio = x[i] / dx[i]
        if ratio < alpha_val:
            alpha_val = ratio
            p_index = i
    return alpha_val, p_index

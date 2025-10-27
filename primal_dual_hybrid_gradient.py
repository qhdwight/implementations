import unittest

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def primal_dual_hybrid_gradient(c: np.ndarray, G: sp.csc_matrix, h: np.ndarray) -> tuple[np.ndarray, float, int]:
    m, n = G.shape
    x = np.zeros(n)
    y = np.zeros(m)

    row_norms = np.maximum(sp.linalg.norm(G, axis=1), 1e-8)
    col_norms = np.maximum(sp.linalg.norm(G, axis=0), 1e-8)

    Dr = sp.diags(1.0 / np.sqrt(row_norms), format='csc')
    Dc = sp.diags(1.0 / np.sqrt(col_norms), format='csc')

    G_scaled = Dr @ G @ Dc
    c_scaled = Dc @ c
    h_scaled = Dr @ h

    sigma_max = sp.linalg.svds(G_scaled, k=1, return_singular_vectors=False).item()
    tau = 0.99 / sigma_max
    sigma = 0.99 / sigma_max
    theta = 1.0

    def proj_non_neg(v):
        return np.maximum(v, 0.0)

    for it in range(1024):
        grad_x = c_scaled + G_scaled.T @ y
        x_new = proj_non_neg(x - tau * grad_x)
        x_bar = x_new + theta * (x_new - x)

        res_y = G_scaled @ x_bar - h_scaled
        y_new = proj_non_neg(y + sigma * res_y)

        pd_gap = np.abs(c_scaled @ x + h_scaled @ y)
        p_feas = np.max(G_scaled @ x - h_scaled)
        d_feas = np.min(G_scaled.T @ y + c_scaled)

        if pd_gap < 1e-8 and p_feas < 1e-8 and -d_feas < 1e-8:
            break

        x, y = x_new, y_new
    else:
        raise ValueError("Maximum iterations exceeded")

    x_orig = Dc @ x
    primal_obj = c @ x_orig

    return x_orig, primal_obj, it + 1


class Tests(unittest.TestCase):
    def test_feasible(self):
        c = np.array([-1, -2])
        G = sp.csc_matrix([
            [+1, +1],
            [-2, -1],
            [-1, -2],
        ])
        h = np.array([+1, -1, -1])
        x, J, it = primal_dual_hybrid_gradient(c, G, h)

        x_cp = cp.Variable(len(c), nonneg=True)
        J_cp = cp.Problem(
            cp.Minimize(c @ x_cp), [G @ x_cp <= h]
        ).solve()

        self.assertTrue(np.isclose(J, J_cp))


if __name__ == '__main__':
    unittest.main()

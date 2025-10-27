import unittest

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from common import line_search


# "Numerical Optimization" by Nocedal and Wright, Section 13.6

def dual_simplex(c: np.ndarray, G: sp.csc_matrix, h: np.ndarray) -> tuple[np.ndarray, float, int]:
    # Solve min c^T x subject to Gx + Is = h, x,s >= 0
    m, n = G.shape
    I = sp.eye(m, format='csc')
    A = sp.hstack([G, I], format='csc')
    k = np.concatenate([c, np.zeros(m)])
    basis = list(range(n, n + m))  # Select s variables as initial basis
    nonbasic = list(range(n))

    for it in range(1024):
        B, N = A[:, basis], A[:, nonbasic]
        yb = sp.linalg.spsolve(B, h)
        cb, cn = k[basis], k[nonbasic]
        lamda = sp.linalg.spsolve(B.T, cb)
        sn = cn - N.T @ lamda
        if np.all(yb >= 0.0):
            y = np.zeros(n + m + m)
            y[basis] = yb
            x, s = y[:n], y[n:n + m]
            if np.any(s < 0.0):
                raise ValueError("Unbounded problem")
            J = np.dot(c, x)
            break
        q_index = np.argmin(yb)
        e_q = np.zeros(m)
        e_q[q_index] = 1.0
        v = sp.linalg.spsolve(-B.T, e_q)
        w = N.T @ v
        if np.all(w <= 0.0):
            raise ValueError("Infeasible problem")
        _alpha, r_index = line_search(sn, w)
        r = nonbasic[r_index]
        q = basis[q_index]
        # Update basis
        basis[q_index] = r
        nonbasic[nonbasic.index(r)] = q
    else:
        raise ValueError("Maximum iterations exceeded")

    return x, J, it + 1


class Tests(unittest.TestCase):
    def test_feasible(self):
        # Solve min c^T x subject to Gx <= h, x >= 0
        c = np.array([-1, -2])
        G = sp.csc_matrix([
            [+1, +1],
            [-2, -1],
            [-1, -2],
        ])
        h = np.array([+1, -1, -1])
        x, J, it = dual_simplex(c, G, h)

        x_cp = cp.Variable(len(c), nonneg=True)
        J_cp = cp.Problem(
            cp.Minimize(c @ x_cp), [G @ x_cp <= h]
        ).solve()

        self.assertTrue(np.isclose(J, J_cp))

        print("Optimal value:", J)
        print("CVXPY Solution:", x_cp.value)
        print("Simplex Solution:", x)
        print("Simplex iterations:", it)


if __name__ == '__main__':
    unittest.main()

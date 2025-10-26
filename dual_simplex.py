import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from common import line_search


# "Numerical Optimization" by Nocedal and Wright, Section 13.6

def main():
    # Solve min c^T x subject to Gx <= h, x >= 0
    c = np.array([-1, -2])
    G = sp.csc_matrix([
        [+1, +1],
        [-2, -1],
        [-1, -2],
    ])
    h = np.array([+1, -1, -1])

    # Phase I: Solve min sum(s) subject to Gx + Is = h, x,s >= 0
    m, n = G.shape
    I = sp.eye(m, format='csc')
    A = sp.hstack([G, I], format='csc')
    k = np.concatenate([np.zeros(n), np.zeros(m)])
    basis = list(range(n, n + m))  # Select s variables as initial basis
    nonbasic = list(range(n))
    # Phase II: Solve min c^T x subject to Gx + Is = h, x,s >= 0

    phase = 0
    for it in range(64):
        B, N = A[:, basis], A[:, nonbasic]
        yb = sp.linalg.spsolve(B, h)
        cb, cn = k[basis], k[nonbasic]
        lam = sp.linalg.spsolve(B.T, cb)
        sn = cn - N.T @ lam
        if phase == 0:
            if np.all(yb >= 0.0):
                k = np.concatenate([c, np.zeros(m)])
                phase = 1
                continue
            p_index = np.argmin(yb)
            e_p = np.zeros(m)
            e_p[p_index] = 1.0
            v = sp.linalg.spsolve(B.T, e_p)
            w = N.T @ v
            _, q_index = line_search(sn, -w)
            if q_index < 0:
                raise ValueError("Infeasible problem")
            q = nonbasic[q_index]
        elif phase == 1:
            if np.all(sn >= 0.0):
                y = np.zeros(n + m)
                y[basis] = yb
                x = y[:n]
                J = np.dot(c, x)
                break
            q_index = np.argmin(sn)
            q = nonbasic[q_index]
            d = sp.linalg.spsolve(B, A[:, q])
            if np.all(d <= 0.0):
                raise ValueError("Problem is unbounded")
            _, p_index = line_search(yb, d)
        else:
            raise ValueError("Invalid phase")
        # Update basis
        old_p = basis[p_index]
        basis[p_index] = q
        nonbasic[nonbasic.index(q)] = old_p
    else:
        raise ValueError("Maximum iterations exceeded")

    x_cp = cp.Variable(n, nonneg=True)
    J_cp = cp.Problem(
        cp.Minimize(c @ x_cp), [G @ x_cp <= h]
    ).solve()

    assert np.isclose(J, J_cp)

    print("Optimal value:", J)
    print("CVXPY Solution:", x_cp.value)
    print("Simplex Solution:", x)
    print("Simplex iterations:", it + 1)


if __name__ == '__main__':
    main()

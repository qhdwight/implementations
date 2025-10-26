import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from common import line_search


# "Numerical Optimization" by Nocedal and Wright, Section 13.3

def main():
    # Solve min c^T x subject to Gx <= h, x >= 0
    c = np.array([-1, -2])
    G = sp.csc_matrix([
        [+1, +1],
        [-2, -1],
        [-1, -2],
    ])
    h = np.array([+1, -1, -1])

    # Phase I: Solve min sum(z) subject to Gx + Is + Ez = h, x,s,z >= 0
    m, n = G.shape
    I = sp.eye(m, format='csc')
    E = sp.diags(np.where(h >= 0.0, +1.0, -1.0), 0, format='csc')
    A = sp.hstack([G, I, E], format='csc')
    k = np.concatenate([np.zeros(n), np.zeros(m), np.ones(m)])
    basis = list(range(n + m, n + m + m))  # Select z variables as initial basis
    nonbasic = list(range(n + m))
    # Phase II: Solve min c^T x subject to Gx + Is + Iz = h, x,s,z >= 0

    phase = 0
    for it in range(64):
        B, N = A[:, basis], A[:, nonbasic]
        yb = sp.linalg.spsolve(B, h)
        assert np.all(yb >= 0.0)
        cb, cn = k[basis], k[nonbasic]
        lam = sp.linalg.spsolve(B.T, cb)
        sn = cn - N.T @ lam
        if np.all(sn >= 0.0):
            y = np.zeros(n + m + m)
            y[basis] = yb
            if phase == 0:
                z = y[n + m:]
                if np.dot(k, y) > 0.0:
                    raise ValueError("Infeasible problem")
                assert np.all(z <= 1e-8)
                A = sp.hstack([G, I, I], format='csc')
                k = np.concatenate([c, np.zeros(m), np.zeros(m)])
                phase = 1
                continue
            elif phase == 1:
                x = y[:n]
                J = np.dot(c, x)
                break
            else:
                raise ValueError("Invalid phase")
        q = nonbasic[np.argmin(sn)]
        d = sp.linalg.spsolve(B, A[:, q])
        if np.all(d <= 0.0):
            raise ValueError("Problem is unbounded")
        _, p_index = line_search(yb, d)
        p = basis[p_index]
        # Update basis
        basis[p_index] = q
        nonbasic[nonbasic.index(q)] = p
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

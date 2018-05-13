import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    k, m = 5, 7
    alpha = 0.003
    err_margin = 0.001

    np.random.seed(101)

    A = np.round(np.random.rand(m, k) * 10)
    b = np.random.randn(m, 1)
    W = np.random.randn(k, m)
    e = np.eye(k)

    ts = [(A[:, row], e[:, row]) for row in range(k)]

    error_history = []
    error = 1
    epoch = 0

    while error > err_margin:
        epoch += 1

        mean_err = []

        for i in range(k):
            err = np.reshape(e[:, i] - W @ A[:, i], (-1, 1))
            error = np.linalg.norm(err)
            mean_err.append(error)
            W += alpha * (err @ np.reshape(A[:, i], (1, -1)))

        if epoch % 10 == 0:
            error_history.append(np.array(mean_err).mean())

    Ap = np.linalg.pinv(A)

    print("Calculated output:\n", np.abs(np.round(W @ A)))

    x = Ap @ b + (e - Ap @ A) @ W
    print("Calculated x:\n", x)

    print("Mean error:", np.mean(W - Ap))

    plt.figure()
    plt.title("Error history")
    plt.xlabel("Epochs")
    plt.ylabel("Mean error")
    plt.plot(error_history)
    plt.show()

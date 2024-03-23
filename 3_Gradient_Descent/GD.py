import numpy as np


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        yp = m_curr * x + b_curr

        # Mean Squared Error
        cost = (1 / n) * sum([val**2 for val in (y - yp)]) 

        # Partial Derivative of m and b
        md = -(2 / n) * sum(x * (y - yp))
        bd = -(2 / n) * sum(y - yp)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {} iterations {}".format(m_curr, b_curr, cost, i))
    pass


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)

# Do try with different learning rate and iterations to find optimal cost
# Compare the cost with its previous value. If there is no major difference, then stop it.
# We get y = 2*x + 3

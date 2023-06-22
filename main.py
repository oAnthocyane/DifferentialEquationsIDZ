import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return (y * y - x * y) / (x * x)


def runge_kutta(h, x0, y0, X):
    N = int((X - x0) / h)
    x = x0
    y = y0
    xs = np.zeros(N + 1)
    ys = np.zeros(N + 1)
    xs[0] = x0
    ys[0] = y0
    for i in range(1, N + 1):
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = x + h
        xs[i] = x
        ys[i] = y
    return xs, ys


with open("in.txt", "r") as file:
    X, h = map(float, file.readlines())

x0, y0 = 0.5, 0.8

x1, y1 = runge_kutta(h, x0, y0, X)
x2, y2 = runge_kutta(h / 10, x0, y0, X)

x3 = np.arange(x0, X, h / 10)
y3 = (2 * x3) / (x3 * x3 + 1)

with open("out.txt", "w") as file:
    for x, y in zip(x1, y1):
        file.write(f"x: {x}, y: {y}\n")

plt.plot(x1, y1, label=f'h = {h}', color='green', marker='o', ms=6)
plt.plot(x2, y2, label=f'h = {h / 10}', color='red', marker='^', ms=4)
plt.plot(x3, y3, label='original', color='blue', marker='')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig("graphic_func.png")
plt.show()

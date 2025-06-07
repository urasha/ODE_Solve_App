import math

ODE_LIST = [
    ("y' = x + y",
     lambda x, y: x + y,
     lambda x, C: C * math.exp(x) - x - 1),
    ("y' = y - x^2 + 1",
     lambda x, y: y - x * x + 1,
     lambda x, C: C * math.exp(x) + x * x + 2 * x + 1),
    ("y' = cos(x) - y",
     lambda x, y: math.cos(x) - y,
     lambda x, C: 0.5 * (math.sin(x) + math.cos(x)) + C * math.exp(-x)),
    ("y' = 2*x*y",
     lambda x, y: 2 * x * y,
     lambda x, C: C * math.exp(x * x)),
]


def euler_method(f, x0, y0, h, N):
    xs = [x0 + i * h for i in range(N + 1)]
    ys = [y0]

    for i in range(1, N + 1):
        xi = xs[i - 1]
        yi = ys[-1]
        ys.append(yi + h * f(xi, yi))

    return xs, ys


def improved_euler_method(f, x0, y0, h, N):
    xs = [x0 + i * h for i in range(N + 1)]
    ys = [y0]

    for i in range(1, N + 1):
        xi, yi = xs[i - 1], ys[-1]
        f1 = f(xi, yi)
        y_pred = yi + h * f1
        f2 = f(xs[i], y_pred)
        ys.append(yi + (h / 2) * (f1 + f2))

    return xs, ys


def milne_method(f, x0, y0, h, N, init_vals):
    xs = [x0 + i * h for i in range(N + 1)]

    if len(init_vals) < 4:
        raise ValueError("Для Milne нужно 4 начальных значения.")

    ys = init_vals[:4].copy()

    for j in range(3, N):
        x_im3, x_im2, x_im1, x_i = xs[j - 3], xs[j - 2], xs[j - 1], xs[j]
        y_im3, y_im2, y_im1, y_i = ys[j - 3], ys[j - 2], ys[j - 1], ys[j]

        # предиктор
        y_pred = y_im3 + (4 * h / 3) * (2 * f(x_i, y_i)
                                        - f(x_im1, y_im1)
                                        + 2 * f(x_im2, y_im2))
        x_new = x_i + h

        # корректор
        y_corr = y_im1 + (h / 3) * (f(x_im1, y_im1)
                                    + 4 * f(x_i, y_i)
                                    + f(x_new, y_pred))
        xs.append(x_new)
        ys.append(y_corr)

    return xs[:N + 1], ys[:N + 1]

import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QComboBox, QVBoxLayout, QHBoxLayout,
    QFormLayout, QTableWidget, QTableWidgetItem, QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from methods.solvers import (
    ODE_LIST,
    euler_method,
    improved_euler_method,
    milne_method
)


class ODESolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Численное решение ОДУ")
        self.setGeometry(100, 100, 800, 600)
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        eq_layout = QHBoxLayout()
        eq_layout.addWidget(QLabel("Выберите уравнение y' = f(x,y):"))
        self.comboEq = QComboBox()
        for label, _, _ in ODE_LIST:
            self.comboEq.addItem(label)
        eq_layout.addWidget(self.comboEq)
        layout.addLayout(eq_layout)

        form = QFormLayout()
        self.x0_edit = QLineEdit();
        form.addRow("x₀:", self.x0_edit)
        self.y0_edit = QLineEdit();
        form.addRow("y₀:", self.y0_edit)
        self.xn_edit = QLineEdit();
        form.addRow("xₙ:", self.xn_edit)
        self.h_edit = QLineEdit();
        form.addRow("Шаг h (рекомендуется ≤0.1):", self.h_edit)
        self.eps_edit = QLineEdit();
        form.addRow("Точность ε:", self.eps_edit)
        layout.addLayout(form)

        self.solve_button = QPushButton("Рассчитать")
        self.solve_button.clicked.connect(self.on_solve)
        layout.addWidget(self.solve_button)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.error_label_euler = QLabel("")
        self.error_label_impeuler = QLabel("")
        self.error_label_milne = QLabel("")
        layout.addWidget(self.error_label_euler)
        layout.addWidget(self.error_label_impeuler)
        layout.addWidget(self.error_label_milne)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def solve_with_runge(self, method, f, x0, y0, xn, h, eps, p):
        MAX_POINTS = 1_000_000
        while True:
            N = int(round((xn - x0) / h))
            if N > MAX_POINTS:
                QMessageBox.critical(self, "Ошибка", "Превышено максимальное число точек.")
                return None, None, None, None, None

            xs1, ys1 = method(f, x0, y0, h, N)
            xs2, ys2 = method(f, x0, y0, h / 2, 2 * N)

            denom = 2 ** p - 1
            err = max(abs(y1 - y2) / denom for y1, y2 in zip(ys1, ys2[::2]))

            if err <= eps:
                return xs1, ys1, h, N, err

            h /= 2

    def on_solve(self):
        try:
            x0 = float(self.x0_edit.text())
            y0 = float(self.y0_edit.text())
            xn = float(self.xn_edit.text())
            h0 = float(self.h_edit.text())
            eps = float(self.eps_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите корректные числовые данные.")
            return

        if not (h0 > 0 and eps > 0 and xn > x0):
            QMessageBox.critical(self, "Ошибка", "Проверьте условия: xₙ > x₀, h > 0, ε > 0.")
            return

        idx = self.comboEq.currentIndex()
        _, f, exact_fun = ODE_LIST[idx]
        if idx == 0:
            C = (y0 + x0 + 1) / math.exp(x0)
        elif idx == 1:
            C = (y0 - (x0 * x0 + 2 * x0 + 1)) / math.exp(x0)
        elif idx == 2:
            C = (y0 - 0.5 * (math.sin(x0) + math.cos(x0))) * math.exp(x0)
        else:
            C = y0 * math.exp(-x0 * x0)

        xs_e, ys_e, h_e, N_e, err_e = self.solve_with_runge(
            euler_method, f, x0, y0, xn, h0, eps, p=1
        )
        if xs_e is None:
            return

        xs_ie, ys_ie, h_ie, N_ie, err_ie = self.solve_with_runge(
            improved_euler_method, f, x0, y0, xn, h0, eps, p=2
        )
        if xs_ie is None:
            return

        h_used = min(h_e, h_ie)
        N_used = int(round((xn - x0) / h_used))

        xs, y_euler = euler_method(f, x0, y0, h_used, N_used)
        _, y_ieuler = improved_euler_method(f, x0, y0, h_used, N_used)
        ys_init = y_ieuler[:4]
        _, y_milne = milne_method(f, x0, h_used, N_used, ys_init)

        y_exact = [exact_fun(x, C) for x in xs]

        self.table.clear()
        cols = ["x", "Точное y", "Эйлер", "Улучш. Эйлер", "Милн"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(xs))
        for i, xi in enumerate(xs):
            self.table.setItem(i, 0, QTableWidgetItem(f"{xi:.6g}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{y_exact[i]:.6g}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{y_euler[i]:.6g}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{y_ieuler[i]:.6g}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{y_milne[i]:.6g}"))

        self.error_label_euler.setText(f"Эйлер:  h={h_used:.3g}, ε≈{err_e:.3g}")
        self.error_label_impeuler.setText(f"Улучш.Эйлер: h={h_used:.3g}, ε≈{err_ie:.3g}")

        diffs = [abs(y_exact[i] - y_milne[i]) for i in range(len(y_milne))]
        err_m = max(diffs) if diffs else None
        self.error_label_milne.setText(f"Милн: εₘₐₓ≈{err_m:.3g}" if err_m is not None else "")

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(xs, y_exact,
                   label="Точное", marker='x', s=100, linewidths=2,
                   color='black', zorder=5)
        ax.plot(xs, y_euler, label="Эйлер", linestyle='--', marker='o')
        ax.plot(xs, y_ieuler, label="Улучш. Эйлер", linestyle='-.', marker='s')
        ax.plot(xs, y_milne, label="Милн", linestyle=':', marker='^')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ODESolverApp()
    win.show()
    sys.exit(app.exec_())

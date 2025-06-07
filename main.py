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
        self.x0_edit = QLineEdit()
        form.addRow("x₀:", self.x0_edit)
        self.y0_edit = QLineEdit()
        form.addRow("y₀:", self.y0_edit)
        self.xn_edit = QLineEdit()
        form.addRow("xₙ:", self.xn_edit)
        self.h_edit = QLineEdit()
        form.addRow("Шаг h:", self.h_edit)
        self.eps_edit = QLineEdit()
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

    def on_solve(self):
        try:
            x0 = float(self.x0_edit.text())
            y0 = float(self.y0_edit.text())
            xn = float(self.xn_edit.text())
            h = float(self.h_edit.text())
            eps = float(self.eps_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите корректные числовые данные.")
            return

        if not (h > 0 and eps > 0 and xn > x0):
            QMessageBox.critical(self, "Ошибка", "Проверьте условия: xₙ > x₀, h > 0, ε > 0.")
            return

        idx = self.comboEq.currentIndex()
        label, f, exact_fun = ODE_LIST[idx]

        if idx == 0:
            C = (y0 + x0 + 1) / math.exp(x0)
        elif idx == 1:
            C = (y0 - (x0 * x0 + 2 * x0 + 1)) / math.exp(x0)
        elif idx == 2:
            C = (y0 - 0.5 * (math.sin(x0) + math.cos(x0))) * math.exp(x0)
        else:
            C = y0 * math.exp(-x0 * x0)

        N = int((xn - x0) // h)

        xs_e, ys_e = euler_method(f, x0, y0, h, N)
        xs_ie, ys_ie = improved_euler_method(f, x0, y0, h, N)
        ys_exact = [exact_fun(x, C) for x in xs_e]
        ys_init = ys_ie[:4]
        xs_m, ys_m = milne_method(f, x0, y0, h, N, ys_init)

        x_vals = xs_e
        y_euler = ys_e
        y_impeuler = ys_ie
        y_exact = ys_exact
        y_milne = ys_m

        # Оценка погрешностей
        try:
            # Эйлер
            y_h = y_euler[-1]
            y_half = y0
            for j in range(1, N * 2 + 1):
                xj = x0 + (j - 1) * (h / 2)
                y_half += (h / 2) * f(xj, y_half)
            eps_e = abs(y_half - y_h) / (2 ** 1 - 1)

            # Усоверш. Эйлер
            y_h2 = y_impeuler[-1]
            y_half2 = y0
            for j in range(1, N * 2 + 1):
                xj = x0 + (j - 1) * (h / 2)
                f1 = f(xj, y_half2)
                tmp = y_half2 + (h / 2) * f1
                f2 = f(xj + h / 2, tmp)
                y_half2 += (h / 2) / 2 * (f1 + f2)
            eps_ie = abs(y_half2 - y_h2) / (2 ** 2 - 1)
        except Exception:
            eps_e = eps_ie = None

        # Максимальная погрешность Милна
        diffs = [abs(y_exact[i] - y_milne[i]) for i in range(len(y_milne))]
        eps_m = max(diffs) if diffs else None

        # Вывод в таблицу
        cols = ["x", "Точное y", "Эйлер", "Улучш. Эйлер", "Милн"]
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(x_vals))
        for i, xv in enumerate(x_vals):
            self.table.setItem(i, 0, QTableWidgetItem(f"{xv:.6g}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{y_exact[i]:.6g}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{y_euler[i]:.6g}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{y_impeuler[i]:.6g}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{y_milne[i]:.6g}"))

        self.error_label_euler.setText(
            f"Погрешность Эйлера: ε≈{eps_e:.3g}" if eps_e is not None else ""
        )
        self.error_label_impeuler.setText(
            f"Погрешность Улучш. Эйлер: ε≈{eps_ie:.3g}" if eps_ie is not None else ""
        )
        self.error_label_milne.setText(
            f"Погрешность Милна: εₘₐₓ={eps_m:.3g}" if eps_m is not None else ""
        )

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(
            x_vals, y_exact,
            label="Точное",
            marker='x', s=100, linewidths=2,
            color='black', zorder=5
        )
        ax.plot(x_vals, y_euler, label="Эйлер", linestyle='--', marker='o')
        ax.plot(x_vals, y_impeuler, label="Улучш. Эйлер", linestyle='-.', marker='s')
        ax.plot(x_vals, y_milne, label="Милн", linestyle=':', marker='^')
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

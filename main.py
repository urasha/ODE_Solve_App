# -*- coding: utf-8 -*-
import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QComboBox, QVBoxLayout, QHBoxLayout,
    QFormLayout, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ODESolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Численное решение ОДУ")
        self.setGeometry(100, 100, 800, 600)
        self._init_ui()

    def _init_ui(self):
        # Основной виджет и компоновка
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Выбор уравнения из списка
        eq_layout = QHBoxLayout()
        eq_layout.addWidget(QLabel("Выберите уравнение y' = f(x,y):"))
        self.comboEq = QComboBox()
        self.equations = []
        def f1(x,y): return x+y
        def sol1(x,C): return C*math.exp(x)-x-1
        self.equations.append({"label":"y' = x + y","func":f1,"exact":sol1})
        def f2(x,y): return y-x*x+1
        def sol2(x,C): return C*math.exp(x)+x*x+2*x+1
        self.equations.append({"label":"y' = y - x^2 + 1","func":f2,"exact":sol2})
        def f3(x,y): return math.cos(x)-y
        def sol3(x,C): return 0.5*(math.sin(x)+math.cos(x))+C*math.exp(-x)
        self.equations.append({"label":"y' = cos(x) - y","func":f3,"exact":sol3})
        def f4(x,y): return 2*x*y
        def sol4(x,C): return C*math.exp(x*x)
        self.equations.append({"label":"y' = 2*x*y","func":f4,"exact":sol4})
        for eq in self.equations:
            self.comboEq.addItem(eq["label"])
        eq_layout.addWidget(self.comboEq)
        main_layout.addLayout(eq_layout)

        # Параметры задачи
        form = QFormLayout()
        self.x0_edit=QLineEdit(); form.addRow("x0:",self.x0_edit)
        self.y0_edit=QLineEdit(); form.addRow("y0:",self.y0_edit)
        self.xn_edit=QLineEdit(); form.addRow("x_n:",self.xn_edit)
        self.h_edit=QLineEdit(); form.addRow("Шаг h (рекомендуется ≤0.1):",self.h_edit)
        self.eps_edit=QLineEdit(); form.addRow("Точность ε:",self.eps_edit)
        main_layout.addLayout(form)

        self.solve_button=QPushButton("Рассчитать")
        self.solve_button.clicked.connect(self.on_solve)
        main_layout.addWidget(self.solve_button)

        self.table=QTableWidget(); main_layout.addWidget(self.table)
        self.error_layout=QVBoxLayout()
        self.error_label_euler=QLabel("")
        self.error_label_impeuler=QLabel("")
        self.error_label_milne=QLabel("")
        self.error_layout.addWidget(self.error_label_euler)
        self.error_layout.addWidget(self.error_label_impeuler)
        self.error_layout.addWidget(self.error_label_milne)
        main_layout.addLayout(self.error_layout)

        self.figure=Figure(figsize=(5,4))
        self.canvas=FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def on_solve(self):
        # Ввод
        try:
            x0=float(self.x0_edit.text()); y0=float(self.y0_edit.text())
            xn=float(self.xn_edit.text()); h=float(self.h_edit.text())
            eps=float(self.eps_edit.text())
        except:
            QMessageBox.critical(self,"Ошибка","Введите корректные числовые данные.")
            return
        if h<=0 or eps<=0 or xn<=x0:
            QMessageBox.critical(self,"Ошибка","Проверьте условия: x_n>x0, h>0, ε>0.")
            return
        idx=self.comboEq.currentIndex()
        eq=self.equations[idx]; f=eq["func"]; exact_fun=eq["exact"]
        # C
        if idx==0: C=(y0+x0+1)/math.exp(x0)
        elif idx==1: C=(y0-(x0*x0+2*x0+1))/math.exp(x0)
        elif idx==2: C=(y0-0.5*(math.sin(x0)+math.cos(x0)))*math.exp(x0)
        else: C=y0*math.exp(-x0*x0)
        # узлы
        N=int((xn-x0)//h)
        x_vals=[x0+i*h for i in range(N+1)]
        # Эйлер и улучш. Эйлер
        y_euler=[y0]; y_impeuler=[y0]; y_exact=[y0]
        for i in range(1,N+1):
            xi=x_vals[i-1]
            # Эйлер
            ye=y_euler[-1]+h*f(xi,y_euler[-1]); y_euler.append(ye)
            # улучшенный
            yi=y_impeuler[-1]; f1=f(xi,yi); tmp=yi+h*f1
            f2=f(x_vals[i],tmp); yim=yi+(h/2)*(f1+f2); y_impeuler.append(yim)
            # точное
            y_exact.append(exact_fun(x_vals[i],C))
        # Метод Милна
        xs_m=x_vals[:4]
        ys_m=[y_impeuler[i] for i in range(4)]  # init with ул. Эйлер
        for j in range(3,N):
            x_im3,x_im2,x_im1,x_i=xs_m[j-3],xs_m[j-2],xs_m[j-1],xs_m[j]
            y_im3,y_im2,y_im1,y_i=ys_m[j-3],ys_m[j-2],ys_m[j-1],ys_m[j]
            # предиктор
            y_pred=y_im3+(4*h/3)*(2*f(x_i,y_i)-f(x_im1,y_im1)+2*f(x_im2,y_im2))
            x_new=x_i+h
            # корректор
            y_corr=y_im1+(h/3)*(f(x_im1,y_im1)+4*f(x_i,y_i)+f(x_new,y_pred))
            xs_m.append(x_new); ys_m.append(y_corr)
        y_milne=ys_m[:N+1]
        # Оценка
        try:
            N2=N*2; y_h=y_euler[-1]; yh=y0
            for j in range(1,N2+1): yh+= (h/2)*f(x0+(j-1)*(h/2),yh)
            eps_e=abs(yh-y_h)/(2**1-1)
            y_h2=y_impeuler[-1]; yh2=y0
            for j in range(1,N2+1):
                xj=x0+(j-1)*(h/2); f1=f(xj,yh2); tmp=yh2+(h/2)*f1
                f2=f(xj+h/2,tmp); yh2+=(h/2)/2*(f1+f2)
            eps_ie=abs(yh2-y_h2)/(2**2-1)
        except:
            eps_e=eps_ie=None
        diffs=[abs(y_exact[i]-y_milne[i]) for i in range(len(y_milne))]
        eps_m=max(diffs) if diffs else None
        # вывод таблицы
        cols=["x","Точное y","Эйлер","Улучш. Эйлер","Милн"]
        self.table.clear(); self.table.setColumnCount(5); self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(x_vals))
        for i,xv in enumerate(x_vals):
            self.table.setItem(i,0,QTableWidgetItem(f"{xv:.6g}"))
            self.table.setItem(i,1,QTableWidgetItem(f"{y_exact[i]:.6g}"))
            self.table.setItem(i,2,QTableWidgetItem(f"{y_euler[i]:.6g}"))
            self.table.setItem(i,3,QTableWidgetItem(f"{y_impeuler[i]:.6g}"))
            self.table.setItem(i,4,QTableWidgetItem(f"{y_milne[i]:.6g}"))
        self.error_label_euler.setText(f"Погрешность Эйлера: ε≈{eps_e:.3g}" if eps_e is not None else "")
        self.error_label_impeuler.setText(f"Погрешность Улучш. Эйлер: ε≈{eps_ie:.3g}" if eps_ie is not None else "")
        self.error_label_milne.setText(f"Погрешность Милна: ε_max={eps_m:.3g}" if eps_m is not None else "")
        # график
        self.figure.clear(); ax=self.figure.add_subplot(111)
        ax.scatter(x_vals,y_exact,label="Точное",marker='x',s=100,linewidths=2,color='black',zorder=5)
        ax.plot(x_vals,y_euler,label="Эйлер",linestyle='--',marker='o')
        ax.plot(x_vals,y_impeuler,label="Улучш.Эйлер",linestyle='-.',marker='s')
        ax.plot(x_vals,y_milne,label="Милн",linestyle=':',marker='^')
        ax.set_xlabel("x");ax.set_ylabel("y");ax.legend();ax.grid(True)
        self.canvas.draw()

if __name__=="__main__":
    app=QApplication(sys.argv);solver=ODESolverApp();solver.show();sys.exit(app.exec_())

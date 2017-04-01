"""
A Qt GUI for SU2k_iTEBD.py

TODO: 
-add statusbar
-plot fidelity while running
-about message

"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Example(QWidget):
    
    def __init__(self):
        super().__init__() # super returns parent of class, here QWidget whose constructor is called.        
        self.initUI()
        
    def initUI(self):
        self.resize(500, 300)
        self.center()
        self.setWindowTitle('Anyonic iTEBD')
        # tooltip: help of tools
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip('A python3 implementation of the iTEBD algorithm for <b>SU2_k anyons</b>')        
        # grid layout
        grid = QGridLayout()
        self.setLayout(grid)
        # list of names to be displayed
        n = 0
        label = QLabel('chi')
        grid.addWidget(label, n, 1)
        self.chiv = QLineEdit('40')
        grid.addWidget(self.chiv, n, 2)
        n += 1
        label = QLabel('iterations')
        grid.addWidget(label, n, 1)
        self.itv = QLineEdit('1000')
        grid.addWidget(self.itv, n, 2)
        n += 1
        label = QLabel('delta t')
        grid.addWidget(label, n, 1)
        self.dtv = QLineEdit('0.01')
        grid.addWidget(self.dtv, n,2)
        n += 1
        label = QLabel('k+1')
        grid.addWidget(label, n, 1)
        self.pv = QLineEdit('4')
        grid.addWidget(self.pv, n, 2)
        n += 1
        label = QLabel('a')
        grid.addWidget(label, n, 1)
        self.av = QLineEdit('1')
        grid.addWidget(self.av, n,2)
        n += 1
        label = QLabel('b')
        grid.addWidget(label, n, 1)
        self.bv = QLineEdit('1')
        grid.addWidget(self.bv, n,2)
        n += 1
        # run button
        run_btn = QPushButton("Run")
        grid.addWidget(run_btn, n, 1, 1, 2) # spans 1 row and 2 cols
        run_btn.setToolTip('Runs the simulation with these parameters')
        run_btn.clicked.connect(self.run_button_clicked)
        # output
        n += 1
        label = QLabel('output')
        grid.addWidget(label, n, 1)
        self.output = QTextEdit('')
        grid.addWidget(self.output, n,2)

        self.show()

    # what happens when the run button is clicked
    def run_button_clicked(self):
        p = self.pv.text()
        a = self.av.text()
        b = self.bv.text()
        chi = self.chiv.text()
        dt = self.dtv.text()
        N = self.itv.text()
        from subprocess import check_output
        out = check_output(["python3", "SU2k_iTEBD.py", 
                            "-p", p,
                            "-N", N,
                            "-D", chi,
                            "-t", dt,
                            "-a", a,
                            "-b", b,
                            ], universal_newlines=True)

        self.output.setText(str(out.rstrip()))

    # center on screen
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    # what happens if close
    def closeEvent(self, event):        
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()   
        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())  

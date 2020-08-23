import sys
from PyQt5.QtWidgets import *
from RVEGen import RVEGeneratorGUI


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.central_widget = QWidget()
        self.setWindowTitle("DRAGen - Tool by IMS")
        self.initUI()

    def initUI(self):
        self.setCentralWidget(self.central_widget)
        vbox = QVBoxLayout(self.central_widget)
        label = QLabel(self)

        rve = QPushButton("Generate RVEs", self)

        rve.clicked.connect(self.buttonClicked)
        self.generator = RVEGeneratorGUI(self)

        vbox.addStretch(1)
        vbox.addWidget(label)
        vbox.addWidget(rve)

    def buttonClicked(self):
        sender = self.sender()
        if sender.text() == "Generate RVEs":
            self.generator.show()
            self.hide()


def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.resize(400, 100)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    window()

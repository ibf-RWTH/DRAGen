import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QSizePolicy, QLabel, QWidget, QHBoxLayout, QVBoxLayout

class GANTrain(QMainWindow):

    def __init__(self, parent=None):
        super(GANTrain,self).__init__(parent)
        self.initUI()

    def initUI(self):
        QMainWindow.setWindowTitle(self,'Lets generate some Awesome RVEs!!')
        QMainWindow.setSizePolicy(self,QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setGeometry(300, 300, 290, 150)
        self.statusBar().showMessage('Training is not yet implemented')
        start = QPushButton('Start', self)
        start.move(30, 50)
        cancel = QPushButton('Cancel', self)
        cancel.move(150,50)
        start.clicked.connect(self.buttonClicked)
        cancel.clicked.connect(self.buttonClicked)

    def closeAndReturn(self):
        self.close()
        self.parent().show()
        self.parent().statusBar().showMessage('Choose an option')

    def buttonClicked(self):
        sender = self.sender()
        if sender.text() == 'Start':
            self.closeAndReturn()
        elif sender.text() == 'Cancel':
            self.closeAndReturn()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GANTrain()
    sys.exit(app.exec_())
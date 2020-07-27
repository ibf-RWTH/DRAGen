import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from GAN import GANTrain
from RVEGen import RVEGeneratorGUI



class StartWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('DRAGen - Tool by IMS')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        vbox = QVBoxLayout(self.central_widget)
        label = QLabel(self)
        pixmap = QPixmap('./figures/FinalRVE.png').scaled(256,256, Qt.KeepAspectRatio, Qt.FastTransformation)
        label.setPixmap(pixmap)

        #self.textedit = QTextEdit()
        gan = QPushButton("Train GAN", self)
        rve = QPushButton("Generate RVEs", self)

        gan.clicked.connect(self.buttonClicked)
        rve.clicked.connect(self.buttonClicked)
        self.gantrain = GANTrain(self)
        self.generator = RVEGeneratorGUI(self)
        self.statusBar().showMessage('Chose an Option')

        vbox.addStretch(1)
        vbox.addWidget(label)
        vbox.addWidget(gan)
        vbox.addWidget(rve)
        #vbox.addWidget(self.textedit)

    def buttonClicked(self):
        sender = self.sender()
        if sender.text() == "Train GAN":
            self.statusBar().showMessage(sender.text() + ' opening Training')
            self.gantrain.show()
            self.hide()

        elif sender.text() == "Generate RVEs":
            self.statusBar().showMessage('opening Generator')
            self.generator.show()
            self.hide()



if __name__ == '__main__':
    qapp = QApplication(sys.argv)
    app = StartWindow()
    app.show()
    sys.exit(qapp.exec_())
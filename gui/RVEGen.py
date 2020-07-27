
from PyQt5.QtWidgets import QMainWindow, QPushButton, QSizePolicy, QLabel, QWidget, QHBoxLayout, QVBoxLayout, \
     QCheckBox, QSpinBox, QDoubleSpinBox, QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread
from RVEProc import ProcessWindow

class RVEGeneratorGUI(QMainWindow):
    def __init__(self, parent=None):
        super(RVEGeneratorGUI,self).__init__(parent)
        self.initUI()

    def initUI(self):
        QMainWindow.setWindowTitle(self,'Lets generate some Awesome RVEs!!')
        QMainWindow.setSizePolicy(self,QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.statusBar().showMessage('You have to choose at least one solver option!')
        vbox = QVBoxLayout(self.central_widget)
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        hbox7 = QHBoxLayout()
        hbox8 = QHBoxLayout()
        hbox9 = QHBoxLayout()

        felabel = QLabel(self)
        fepixmap = QPixmap('./figures/FinalRVE.png').scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)
        felabel.setPixmap(fepixmap)
        spectrallabel = QLabel(self)
        spectralpixmap = QPixmap('./figures/FinalRVE.png').scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)
        spectrallabel.setPixmap(spectralpixmap)


        self.ferve = QCheckBox('Create RVE for FE-Solver')
        self.specrve = QCheckBox('Create RVE for Spectral-Solver')
        start = QPushButton('Start', self)
        cancel = QPushButton('Cancel', self)

        self.ferve.toggled.connect(self.checkBoxState)
        self.specrve.toggled.connect(self.checkBoxState)
        boxsizelabel = QLabel(self)
        boxsizelabel.setText('Enter Boxsize')
        self.boxsizeinput = QSpinBox(self)
        self.boxsizeinput.setValue(50)
        pointslabel = QLabel(self)
        pointslabel.setText('Number of points on edge: ')
        pointsinput = QSpinBox(self)
        pointsinput.setValue(50)
        pointsinput.setRange(10,512)
        packinglabel = QLabel(self)
        packinglabel.setText('Packingratio of RSA: ')
        packinginput = QDoubleSpinBox(self)
        packinginput.setValue(0.5)
        packinginput.setRange(0.2,0.7)
        packinginput.setSingleStep(0.1)
        bandslabel = QLabel(self)
        bandslabel.setText('Number of Bands: ')
        bandsinput = QSpinBox(self)
        bandsinput.setValue(0)
        widthlabel = QLabel(self)
        widthlabel.setText('width of bands')
        widthinput = QDoubleSpinBox(self)
        widthinput.setValue(0.)
        widthinput.setRange(1.0, 15.0)
        widthinput.setSingleStep(0.1)
        speedlabel = QLabel(self)
        speedlabel.setText('Growing speed for graingrowth: ')
        speedinput = QSpinBox(self)
        speedinput.setValue(1)
        speedinput.setMaximum(10)
        start.clicked.connect(self.buttonClicked)
        cancel.clicked.connect(self.buttonClicked)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addLayout(hbox6)
        vbox.addLayout(hbox7)
        vbox.addLayout(hbox8)
        vbox.addLayout(hbox9)
        #vbox.addLayout(hbox10)

        self.central_widget.setLayout(vbox)
        hbox1.addWidget(felabel)
        hbox1.addWidget(spectrallabel)
        hbox2.addWidget(self.ferve)
        hbox2.addWidget(self.specrve)
        hbox3.addWidget(boxsizelabel)
        hbox3.addWidget(self.boxsizeinput)
        hbox4.addWidget(pointslabel)
        hbox4.addWidget(pointsinput)
        hbox5.addWidget(packinglabel)
        hbox5.addWidget(packinginput)
        hbox6.addWidget(bandslabel)
        hbox6.addWidget(bandsinput)
        hbox7.addWidget(widthlabel)
        hbox7.addWidget(widthinput)
        hbox8.addWidget(speedlabel)
        hbox8.addWidget(speedinput)
        hbox9.addWidget(start)
        hbox9.addWidget(cancel)


        self.rveproc = ProcessWindow(self)

    def closeAndReturn(self):
        self.close()
        self.parent().show()
        self.parent().statusBar().showMessage('Choose an option')

    def buttonClicked(self):
        sender = self.sender()

        if sender.text() == 'Start':
            self.rveproc.show()

        elif sender.text() == 'Cancel':
            self.closeAndReturn()

    def checkBoxState(self):
        ferve = self.ferve
        specrve = self.specrve
        if ferve.isChecked() and not specrve.isChecked():
            self.statusBar().showMessage('FE RVE will be created')
        elif ferve.isChecked() and specrve.isChecked():
            self.statusBar().showMessage('Both versions of RVEs will be created')
        elif not ferve.isChecked() and not specrve.isChecked():
            self.statusBar().showMessage('You have to choose at least one solver option!')
        else:
            self.statusBar().showMessage('Spectral RVE will be created')


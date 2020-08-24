from PyQt5.QtWidgets import *
from gui.RVEProc import ProcessWindow


class RVEGeneratorGUI(QMainWindow):
    def __init__(self, parent=None):
        super(RVEGeneratorGUI, self).__init__(parent)
        self.central_widget = QWidget()
        self.setFixedSize(640, 480)
        self.initUI()

    def initUI(self):
        QMainWindow.setWindowTitle(self, 'RVE Generator')
        self.setCentralWidget(self.central_widget)
        vbox = QVBoxLayout(self.central_widget)
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        hbox7 = QHBoxLayout()
        hbox8 = QHBoxLayout()

        start = QPushButton('Start', self)
        cancel = QPushButton('Cancel', self)

        box_size_label = QLabel(self)
        box_size_label.setText('Boxsize: ')
        self.box_size_input = QSpinBox(self)
        self.box_size_input.setValue(50)
        points_label = QLabel(self)
        points_label.setText('Number of points on edge: ')
        self.points_input = QSpinBox(self)
        self.points_input.setValue(50)
        self.points_input.setRange(10, 512)
        self.points_input.setSingleStep(2)
        packing_label = QLabel(self)
        packing_label.setText('Packing ratio: ')
        self.packing_input = QDoubleSpinBox(self)
        self.packing_input.setValue(0.5)
        self.packing_input.setRange(0.2, 0.7)
        self.packing_input.setSingleStep(0.1)
        bands_label = QLabel(self)
        bands_label.setText('Number of bands: ')
        self.bands_input = QSpinBox(self)
        self.bands_input.setValue(0)
        width_label = QLabel(self)
        width_label.setText('Bandwidth: ')
        self.width_input = QDoubleSpinBox(self)
        self.width_input.setValue(3.)
        self.width_input.setRange(1.0, 15.0)
        self.width_input.setSingleStep(0.1)
        speed_label = QLabel(self)
        speed_label.setText('Growing speed for grain growth: ')
        self.speed_input = QSpinBox(self)
        self.speed_input.setValue(1)
        self.speed_input.setMaximum(10)
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

        self.central_widget.setLayout(vbox)
        hbox1.addWidget(box_size_label)
        hbox1.addWidget(self.box_size_input)
        hbox2.addWidget(points_label)
        hbox2.addWidget(self.points_input)
        hbox3.addWidget(packing_label)
        hbox3.addWidget(self.packing_input)
        hbox4.addWidget(bands_label)
        hbox4.addWidget(self.bands_input)
        hbox5.addWidget(width_label)
        hbox5.addWidget(self.width_input)
        hbox6.addWidget(speed_label)
        hbox6.addWidget(self.speed_input)
        hbox7.addWidget(start)
        hbox8.addWidget(cancel)

        self.rveproc = ProcessWindow(self.box_size_input.value(), self.points_input.value(), self.packing_input.value(),
                                     self.bands_input.value(), self.width_input.value(), self.speed_input.value())

    def closeAndReturn(self):
        self.close()
        self.parent().show()

    def buttonClicked(self):
        sender = self.sender()
        #last_RVE = 3

        if sender.text() == 'Start':
            if self.points_input.value() % 2 == 0:
                self.statusBar().clearMessage()
                self.rveproc.show()
                #self.obj = DataTask(self.box_size_input.value(), self.points_input.value(),
                 #                   self.bands_input.value(), self.width_input.value(), self.speed_input.value())
                #convert_list, phase1, phase2 = self.obj.initializations
                #for i in range(last_RVE + 1):
                    # self.obj.rve_generation(i, convert_list, phase1, phase2)
                 #   self.statusBar().showMessage('Iteration ' + str(i) + ': RVE generation in progress...')
            else:
                self.statusBar().showMessage('Number of points on edge input has to be even.')

        elif sender.text() == 'Cancel':
            self.closeAndReturn()

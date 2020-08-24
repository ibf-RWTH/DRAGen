from PyQt5.QtWidgets import *
from gui.testprint import Generator


class ProcessWindow(QMainWindow):
    def __init__(self, box_size, points_on_edge, packing_ratio, bands, bandwidth, speed, parent=None):
        super(ProcessWindow, self).__init__(parent)
        self.worker_thread = Generator()
        self.setFixedSize(540, 280)
        self.worker_thread.job_done.connect(self.on_job_done)
        self.box_size = box_size
        self.points_on_edge = points_on_edge
        self.packing_ratio = packing_ratio
        self.bands = bands
        self.bandwidth = bandwidth
        self.speed = speed
        self.initUI()

    def initUI(self):
        self.setWindowTitle('RVE Generator')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        vbox = QVBoxLayout(self.central_widget)

        self.textedit = QTextEdit("Result:")
        self.start = QPushButton("Start", self)
        self.cancel = QPushButton("Cancel", self)

        self.start.clicked.connect(self.start_thread)
        self.cancel.clicked.connect(self.buttonClicked)

        vbox.addStretch(1)
        vbox.addWidget(self.textedit)
        vbox.addWidget(self.start)
        vbox.addWidget(self.cancel)

    def start_thread(self):
        self.worker_thread.gui_text = self.textedit.setText('')
        self.worker_thread.start()

    def on_job_done(self, generated_str):
        self.textedit.append(generated_str)

    def closeAndReturn(self):
        self.close()
        self.parent().show()

    def buttonClicked(self):
        sender = self.sender()
        if sender.text() == "Start":
            pass
        elif sender.text() == "Cancel":
            self.closeAndReturn()

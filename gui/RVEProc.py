from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from testprint import Generator


class ProcessWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ProcessWindow,self).__init__(parent)
        self.worker_thread = Generator()
        self.worker_thread.job_done.connect(self.on_job_done)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DRAGen - Tool by IMS')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        vbox = QVBoxLayout(self.central_widget)

        self.textedit = QTextEdit("result:")
        self.start = QPushButton("Start",self)
        self.cancel = QPushButton("Cancel",self)

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
        print("Generated string : ", generated_str)
        self.textedit.append(generated_str)

    def closeAndReturn(self):
        self.close()
        self.parent().show()
        self.parent().statusBar().showMessage('Choose an option')

    def buttonClicked(self):
        sender = self.sender()
        if sender.text() == "Start":
            pass


        elif sender.text() == "Cancel":
            self.closeAndReturn()



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = ProcessWindow()
    w.show()
    sys.exit(app.exec_())
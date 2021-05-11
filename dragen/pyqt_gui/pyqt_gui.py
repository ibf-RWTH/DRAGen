# Filename: main_window.py

"""Main Window-Style application."""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QIcon
from dragen.pyqt_gui.ScrollLabel import ScrollLabel

from dragen.pyqt_gui.worker import Worker


class Window(QMainWindow, QFileDialog):

    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super(Window, self).__init__(parent)

        self.setWindowTitle('DRAGen')


        print(sys.argv[0][-3:])
        if sys.argv[0][-3:] == 'exe':  # current file location is path + DRAGen.exe
            self.thumbnail_path = sys.argv[0][:-11] + "\\thumbnails\\"
        else:  # current file location is path + pyqt_gui.py
            self.thumbnail_path = sys.argv[0][:-12] + "\\..\\thumbnails\\"
            print(self.thumbnail_path)

        self.setWindowIcon(QIcon(self.thumbnail_path + '\\Drache.ico'))
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self._initUI()

    def _initUI(self):
        self._createMenu()
        self._createToolBar()
        self._createStatusBar()
        box_size_label = QLabel('RVE-Size (µm)')
        self.box_size_Edit = QSpinBox()
        self.box_size_Edit.setMinimum(10)
        self.box_size_Edit.setValue(20)
        self.box_size_Edit.valueChanged.connect(self.bandwidth_handler)

        resolution_label = QLabel('Resolution')
        self.resolution_Edit = QDoubleSpinBox()
        self.resolution_Edit.setMinimum(1.5)
        self.resolution_Edit.setSingleStep(0.1)

        n_rves_label = QLabel('Number of RVEs')
        self.n_rves_Edit = QSpinBox()
        self.n_rves_Edit.setMinimum(1)
        self.n_rves_Edit.setSingleStep(1)

        n_bands_label = QLabel('Number of Bands')
        self.n_bands_Edit = QSpinBox()
        self.n_bands_Edit.setMinimum(0)

        band_width_label = QLabel('Band Thickness (µm)')
        self.band_width_Edit = QDoubleSpinBox()
        self.band_width_Edit.setMinimum(1)
        self.band_width_Edit.setMaximum(self.box_size_Edit.value() / 10)
        self.band_width_Edit.setSingleStep(0.1)

        self.filepath_phase1 = None
        phase1_label = QLabel('csv-file phase1')
        self.phase1_text_Edit = QLineEdit()

        phase1_button = QPushButton()
        phase1_button.setIcon(QIcon(QPixmap(self.thumbnail_path + "\\folder.png")))
        phase1_button.clicked.connect(self.phase1_button_handler)

        self.phase_ratio = None
        phase_ratio_label = QLabel('phase ratio of phase 1')
        self.phase_ratio_Edit = QDoubleSpinBox()
        self.phase_ratio_Edit.setMinimum(0.1)
        self.phase_ratio_Edit.setMaximum(1.0)
        self.phase_ratio_Edit.setValue(1.0)
        self.phase_ratio_Edit.setSingleStep(0.1)


        self.filepath_phase2 = None
        phase2_label = QLabel('csv-file phase2')
        self.phase2_text_Edit = QLineEdit()

        phase2_button = QPushButton()
        phase2_button.setIcon(QIcon(QPixmap(self.thumbnail_path + "\\folder.png")))
        phase2_button.clicked.connect(self.phase2_button_handler)

        self.save_files = None
        save_files_label = QLabel('save files at:')
        self.save_files_Edit = QLineEdit()
        self.save_files_Edit.setText("C:/temp")

        save_button = QPushButton()
        save_button.setIcon(QIcon(QPixmap(self.thumbnail_path + "\\folder.png")))
        save_button.clicked.connect(self.save_button_handler)

        self.twoDcheckBox = QCheckBox()
        twoDcheckBox_label = QLabel('2D - RVE')
        #self.twoDcheckBox.setChecked(False)
        self.twoDcheckBox.stateChanged.connect(self.dimension_handler)

        self.threeDcheckBox = QCheckBox()
        threeDcheckBox_label = QLabel('3D - RVE')
        #self.threeDcheckBox.setChecked(True)
        self.threeDcheckBox.stateChanged.connect(self.dimension_handler)

        self.visualization = QCheckBox()
        visualization_label = QLabel('Plot Generation Process')
        self.threeDcheckBox.setChecked(False)

        self.Logo = QPixmap(self.thumbnail_path + "\\Logo.png")
        self.Logo = self.Logo.scaled(self.Logo.size()/2)
        self.Logo_label = QLabel()
        self.Logo_label.setPixmap(self.Logo)
        self.Logo_label.resize(1,1)


        # input grid
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(box_size_label, 1, 0)
        grid.addWidget(self.box_size_Edit, 1, 1)

        grid.addWidget(resolution_label, 2, 0)
        grid.addWidget(self.resolution_Edit, 2, 1)

        grid.addWidget(n_rves_label, 3, 0)
        grid.addWidget(self.n_rves_Edit, 3, 1)

        grid.addWidget(n_bands_label, 4, 0)
        grid.addWidget(self.n_bands_Edit, 4, 1)

        grid.addWidget(band_width_label, 5, 0)
        grid.addWidget(self.band_width_Edit, 5, 1)

        grid.addWidget(phase1_label, 6, 0)
        grid.addWidget(self.phase1_text_Edit, 6, 1, 1, 2)
        grid.addWidget(phase1_button, 6, 3)

        grid.addWidget(phase_ratio_label, 7, 0)
        grid.addWidget(self.phase_ratio_Edit, 7, 1)

        grid.addWidget(phase2_label, 8, 0)
        grid.addWidget(self.phase2_text_Edit, 8, 1, 1, 2)
        grid.addWidget(phase2_button, 8, 3)

        grid.addWidget(save_files_label, 9, 0)
        grid.addWidget(self.save_files_Edit, 9, 1, 1, 2)
        grid.addWidget(save_button, 9, 3)

        # check boxes
        grid.addWidget(twoDcheckBox_label, 1, 2)
        grid.addWidget(self.twoDcheckBox, 1, 3)
        self.twoDcheckBox.setDisabled(True) # TODO set to disabled while 2D branch still not ready

        grid.addWidget(threeDcheckBox_label, 2, 2)
        grid.addWidget(self.threeDcheckBox, 2, 3)

        grid.addWidget(visualization_label, 3, 2)
        grid.addWidget(self.visualization, 3, 3)

        grid.addWidget(self.Logo_label, 1, 4, 12, 3)
        init_text = """Generation has not yet started"""

        # creating scroll label
        self.info_box = ScrollLabel(self)

        # setting text to the label
        self.info_box.set_text(init_text)
        grid.addWidget(self.info_box, 10, 0, 2, 4)

        # progressbar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(10, 520, 350, 25)
        self.progress.setMaximum(100)

        widget = QWidget()
        widget.setLayout(grid)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Exit', self.close)
        self.menu.addAction('&Import microstructure', self.getfiles)
        self.menu.addAction('&save files', self.save_files)

    def _createToolBar(self):
        self.tools = QToolBar()
        self.addToolBar(self.tools)
        self.tools.addAction('Exit', self.close)
        self.action_submit = QAction('submit', self.tools)
        self.action_import_phase = QAction('import phase data', self.tools)
        self.action_save_data = QAction('save data', self.tools)
        self.tools.addAction(self.action_submit)
        self.tools.addAction(self.action_import_phase)
        self.tools.addAction(self.action_save_data)
        self.action_submit.triggered.connect(self.submit)
        self.action_import_phase.triggered.connect(self.getfiles)
        self.action_save_data.triggered.connect(self.save_files)

    def _createStatusBar(self):
        self.status = QStatusBar()
        self.status.showMessage("Please enter the required information")
        self.setStatusBar(self.status)

    def progress_event(self, progress):
        self.progress.setValue(progress)

    def bandwidth_handler(self):
        self.band_width_Edit.setMaximum(self.box_size_Edit.value()/10)

    def phase1_button_handler(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            self.filepath_phase1 = dlg.selectedFiles()
            self.phase1_text_Edit.setText(self.filepath_phase1[0])

    def phase2_button_handler(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            self.filepath_phase2 = dlg.selectedFiles()
            self.phase2_text_Edit.setText(self.filepath_phase2[0])

    def save_button_handler(self):
        self.save_files = QFileDialog.getExistingDirectory(self)
        self.save_files_Edit.setText(str(self.save_files))

    def dimension_handler(self, state):
        # checking if state is checked
        if state == Qt.Checked:

            # if first check box is selected
            if self.sender() == self.twoDcheckBox:
                # making other check box to uncheck
                self.threeDcheckBox.setChecked(False)
            elif self.sender() == self.threeDcheckBox:
                self.twoDcheckBox.setChecked(False)

    def getfiles(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            if self.filepath_phase1 is None:
                self.filepath_phase1 = dlg.selectedFiles()
                self.phase1_text_Edit.setText(self.filepath_phase1[0])
            else:
                self.filepath_phase2 = dlg.selectedFiles()
                self.phase2_text_Edit.setText(self.filepath_phase2[0])

    def save_files(self):
        self.save_files = QFileDialog.getExistingDirectory(self)
        self.save_files_Edit.setText(str(self.save_files))

    def submit(self):


        box_size = self.box_size_Edit.value()
        resolution = self.resolution_Edit.value()
        n_rves = self.n_rves_Edit.value()
        n_bands = self.n_bands_Edit.value()
        band_width = self.band_width_Edit.value()
        phase1_path = self.phase1_text_Edit.text()
        phase2_path = self.phase2_text_Edit.text()
        phase_ratio = self.phase_ratio_Edit.value()
        store_path = self.save_files_Edit.text()

        store_path_flag = False
        import_flag = False
        dimension = None
        visualization_flag = False
        dimension_flag = False
        gan_flag = False

        if len(phase1_path) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No microstructure was imported")
            msg.setInformativeText("Please import a microstructure file of the following type:\n .pkl, .csv")
            msg.setWindowTitle("Error")
            msg.setDetailedText("input data file is missing")
            msg.exec_()
            return

        if len(phase2_path) == 0:
            phase2_path = None

        if phase1_path[-4:] == '.pkl':
            gan_flag = True
            self.infobox_obj.set_text("microstructure imported from:\n{}".format(self.file1))
            if self.file2 is not None:
                self.info_box.add_text("and from:\n{}".format(phase1_path))
        elif phase1_path[-4:] == '.csv':
            gan_flag = False
            self.info_box.set_text("microstructure imported from\n{}".format(phase1_path))
            if phase2_path is not None:
                self.info_box.add_text("and from\n{}".format(phase2_path))
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Wrong File Type")
            msg.setInformativeText("Your imported microstructure file must be of the following type:\n .pkl, .csv")
            msg.setWindowTitle("Error")
            msg.setDetailedText("The file you imported was of the type: {}".format(str(phase1_path[-4:])))
            msg.exec_()
            return

        if self.twoDcheckBox.isChecked():
            dimension = 2
            dimension_flag = True
        elif self.threeDcheckBox.isChecked():
            dimension = 3
            dimension_flag = True
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please choose dimensionality before starting the generation")
            msg.setInformativeText("Check one of the checkboxes stating\n 2D - RVE or 3D - RVE")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        if len(store_path) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please choose a directory to store the outputdata in")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        if self.visualization.isChecked():
            visualization_flag = True

        if phase1_path is not None:
            import_flag = True

        if phase2_path is not None and phase_ratio == 1:
            msg = QMessageBox()
            reply = msg.question(self,'Warning','The second phase file you chose will not be considered in the RVE.\n'
                                 'Are you sure that you want to keep the phase_ratio at 1.0?', msg.Yes | msg.No)
            if reply == msg.Yes:
                pass
            else:
                return

        if len(store_path) > 0:
            store_path_flag = True
        if dimension_flag and store_path_flag and import_flag:
            self.info_box.add_text('Staring the generation of {} RVEs'.format(n_rves))
            self.thread = QThread()
            self.worker = Worker(box_size, resolution, n_rves, n_bands, band_width, dimension, visualization_flag,
                                 phase1_path, phase2_path, phase_ratio, store_path, gui_flag=True, gan_flag=gan_flag)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.progress_event)
            self.worker.info_box.connect(self.info_box.add_text)
            self.thread.start()
            self.action_submit.setEnabled(False)
            self.status.showMessage('generation running...')
            self.thread.finished.connect(lambda: self.action_submit.setEnabled(True))
            self.thread.finished.connect(lambda: self.status.showMessage('The generation has finished!'))



class ThreadClass(QThread):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)

    def Run(self):
        print('hello')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

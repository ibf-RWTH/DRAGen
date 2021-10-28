# Filename: main_window.py

"""Main Window-Style application."""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QIcon
from dragen.pyqt_gui.ScrollLabel import ScrollLabel

from dragen.pyqt_gui.worker import Worker

ARGS = dict()
ARGS['subs_flag'] = False
ARGS['equiv_d'] = None
ARGS['p_sigma'] = 0.1
ARGS['t_mu'] = None
ARGS['b_sigma'] = 0.1
ARGS['decreasing_factor'] = 0.95
ARGS['lower'] = None
ARGS['upper'] = None
ARGS['circularity'] = 1
ARGS['save'] = True
ARGS['filename'] = 'substruct_data.csv'
ARGS['subs_file_flag'] = False
ARGS['subs_file'] = None
ARGS['subs_flag'] = False
ARGS['phases'] = ['martensite']
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
        box_size_label = QLabel('RVE-Size-X (µm)')
        self.box_size_Edit = QSpinBox()
        self.box_size_Edit.setMinimum(10)
        self.box_size_Edit.setValue(20)
        self.box_size_Edit.valueChanged.connect(self.bandwidth_handler)

        box_size_y_label = QLabel('RVE-Size-Y (µm)')
        self.box_size_y_Edit = QSpinBox()
        self.box_size_y_Edit.setMinimum(10)
        self.box_size_y_Edit.setValue(20)
        self.box_size_y_Edit.valueChanged.connect(self.bandwidth_handler)
        self.box_size_y_Edit.setEnabled(False)

        self.box_size_y_check = QCheckBox()
        self.box_size_y_check.setChecked(False)
        self.box_size_y_check.stateChanged.connect(self.box_size_handler)

        box_size_z_label = QLabel('RVE-Size-Z (µm)')
        self.box_size_z_Edit = QSpinBox()
        self.box_size_z_Edit.setMinimum(10)
        self.box_size_z_Edit.setValue(20)
        self.box_size_z_Edit.valueChanged.connect(self.bandwidth_handler)
        self.box_size_z_Edit.setEnabled(False)

        self.box_size_z_check = QCheckBox()
        self.box_size_z_check.setChecked(False)
        self.box_size_z_check.stateChanged.connect(self.box_size_handler)


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
        phase1_label = QLabel('Ferrite')
        self.phase1_checkBox = QCheckBox()
        self.phase1_checkBox.stateChanged.connect(self.phase_handler)
        self.phase1_text_label  = QLabel('csv-file')
        self.phase1_text_Edit = QLineEdit()
        self.phase1_text_Edit.setFixedWidth(200)
        self.phase1_text_Edit.setEnabled(False)

        self.phase1_button = QPushButton()
        self.phase1_button.setIcon(QIcon(QPixmap(self.thumbnail_path + "\\Folder-Generic-Silver-icon.png")))
        #self.phase1_button.setText('...')
        #self.phase1_button.setFixedWidth(15)
        self.phase1_button.setFixedWidth(50)
        self.phase1_button.clicked.connect(self.button_handler)

        self.phase1_ratio = None
        phase1_ratio_label = QLabel('Ferrite phase ratio')
        self.phase1_ratio_Edit = QDoubleSpinBox()
        self.phase1_ratio_Edit.setMinimum(0.0)
        self.phase1_ratio_Edit.setMaximum(1.0)
        self.phase1_ratio_Edit.setValue(1.0)
        self.phase1_ratio_Edit.setSingleStep(0.1)
        self.phase1_ratio_Edit.setEnabled(False)
        self.phase1_ratio_Edit.valueChanged.connect(self.phase_ratio_handler)

        self.filepath_phase2 = None
        phase2_label = QLabel('Martensite')
        self.phase2_checkBox = QCheckBox()
        self.phase2_checkBox.stateChanged.connect(self.phase_handler)
        self.phase2_text_Edit = QLineEdit()
        self.phase2_text_Edit.setEnabled(False)
        self.phase2_text_label = QLabel('csv-file')
        self.phase2_text_Edit.setFixedWidth(200)

        self.phase2_button = QPushButton()
        self.phase2_button.setIcon(QIcon(QPixmap(self.thumbnail_path + "\\Folder-Generic-Silver-icon.png")))
        self.phase2_button.setFixedWidth(50)
        self.phase2_button.clicked.connect(self.button_handler)

        self.phase2_ratio = None
        phase2_ratio_label = QLabel('Martensite phase ratio')
        self.phase2_ratio_Edit = QDoubleSpinBox()
        self.phase2_ratio_Edit.setMinimum(0.0)
        self.phase2_ratio_Edit.setMaximum(1.0)
        self.phase2_ratio_Edit.setValue(0.0)
        self.phase2_ratio_Edit.setSingleStep(0.1)
        self.phase2_ratio_Edit.setEnabled(False)
        self.phase2_ratio_Edit.valueChanged.connect(self.phase_ratio_handler)

        self.save_files = None
        save_files_label = QLabel('save files at:')
        self.save_files_Edit = QLineEdit()
        self.save_files_Edit.setText("C:/temp")

        self.save_button = QPushButton()
        self.save_button.setIcon(QIcon(QPixmap(self.thumbnail_path + "\\Folder-Generic-Silver-icon.png")))
        self.save_button.setFixedWidth(50)
        self.save_button.clicked.connect(self.button_handler)

        self.twoDcheckBox = QCheckBox()
        twoDcheckBox_label = QLabel('2D - RVE')
        #self.twoDcheckBox.setChecked(False)
        self.twoDcheckBox.stateChanged.connect(self.dimension_handler)

        self.threeDcheckBox = QCheckBox()
        threeDcheckBox_label = QLabel('3D - RVE')
        #self.threeDcheckBox.setChecked(True)
        self.threeDcheckBox.stateChanged.connect(self.dimension_handler)

        self.visualization = QCheckBox()
        visualization_label = QLabel('Visualization Generation Process')
        visualization_label.adjustSize()
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

        grid.addWidget(box_size_y_label,2,0)
        grid.addWidget(self.box_size_y_Edit, 2, 1)
        grid.addWidget(self.box_size_y_check,2,2)

        grid.addWidget(box_size_z_label,3,0)
        grid.addWidget(self.box_size_z_Edit, 3, 1)
        grid.addWidget(self.box_size_z_check,3,2)

        grid.addWidget(resolution_label, 4, 0)
        grid.addWidget(self.resolution_Edit, 4, 1)

        grid.addWidget(n_rves_label, 5, 0)
        grid.addWidget(self.n_rves_Edit, 5, 1)

        grid.addWidget(n_bands_label, 6, 0)
        grid.addWidget(self.n_bands_Edit, 6, 1)

        grid.addWidget(band_width_label, 7, 0)
        grid.addWidget(self.band_width_Edit, 7, 1)

        grid.addWidget(phase1_label, 8, 0)
        grid.addWidget(self.phase1_checkBox,8,1)
        grid.addWidget(self.phase1_text_label,8,2)
        grid.addWidget(self.phase1_text_Edit, 8, 3, 1, 2)
        grid.addWidget(self.phase1_button, 8, 4)

        grid.addWidget(phase1_ratio_label, 9, 0)
        grid.addWidget(self.phase1_ratio_Edit, 9, 1)

        grid.addWidget(phase2_label, 10, 0)
        grid.addWidget(self.phase2_checkBox,10,1)
        grid.addWidget(self.phase2_text_label,10,2)
        grid.addWidget(self.phase2_text_Edit, 10, 3, 1, 2)
        grid.addWidget(self.phase2_button, 10, 4)

        grid.addWidget(phase2_ratio_label, 11, 0)
        grid.addWidget(self.phase2_ratio_Edit, 11, 1)

        grid.addWidget(save_files_label,12, 0)
        grid.addWidget(self.save_files_Edit, 12, 1, 1, 2)
        grid.addWidget(self.save_button, 12, 3)

        # check boxes
        grid.addWidget(twoDcheckBox_label, 1, 3, alignment=Qt.AlignRight)
        grid.addWidget(self.twoDcheckBox, 1, 4, alignment=Qt.AlignRight)
        self.twoDcheckBox.setDisabled(True) # TODO set to disabled while 2D branch still not ready

        grid.addWidget(threeDcheckBox_label, 2, 3, alignment=Qt.AlignRight)
        grid.addWidget(self.threeDcheckBox, 2, 4, alignment=Qt.AlignRight)

        grid.addWidget(visualization_label, 3, 3, alignment=Qt.AlignRight)
        grid.addWidget(self.visualization, 3, 4, alignment=Qt.AlignRight)

        grid.addWidget(self.Logo_label, 1, 5, 12, 3)
        init_text = """Generation has not yet started"""

        # creating scroll label
        self.info_box = ScrollLabel(self)
        # setting text to the label
        self.info_box.set_text(init_text)
        grid.addWidget(self.info_box, 13, 0, 1, 4)

        # progressbar
        self.progress = QProgressBar(self)
        #self.progress.setGeometry(10, 520, 350, 25)
        self.progress.setMaximum(100)
        grid.addWidget(self.progress,14,0,1,4)

        widget = QWidget()
        widget.setLayout(grid)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Exit', self.close)
        self.menu.addAction('&Import phase data', self.getfiles)
        self.menu.addAction('&Save files', self.save_files)

    def _createToolBar(self):
        self.tools = QToolBar()
        self.addToolBar(self.tools)
        self.tools.addAction('&Exit', self.close)
        self.action_submit = QAction('&Submit', self.tools)
        self.action_import_phase = QAction('Import phase data', self.tools)
        self.action_gen_substructure = QToolButton()
        self.action_gen_substructure.setText("Generate substructure")
        self.action_gen_substructure.setPopupMode(QToolButton.MenuButtonPopup)
        subs_menu = QMenu()
        subs_menu.addAction("user_define",self.gen_user_define)
        subs_menu.addAction("from_file",self.gen_from_file)
        self.action_gen_substructure.setMenu(subs_menu)
        self.action_save_data = QAction('Save data', self.tools)

        self.tools.addAction(self.action_import_phase)
        # self.tools.addAction(self.action_gen_substructure)
        self.tools.addWidget(self.action_gen_substructure)
        self.tools.addAction(self.action_save_data)
        self.tools.addAction(self.action_submit)

        self.action_import_phase.triggered.connect(self.getfiles)
        # self.action_gen_substructure.triggered.connect(self.subs_combo)
        self.action_save_data.triggered.connect(self.save_files)
        self.action_submit.triggered.connect(self.submit)

    def gen_from_file(self):
        file_win = FromFileWindow()
        file_win.show()
        file_win.exec_()

    def gen_user_define(self):
        user_win = UserDefineWindow()
        user_win.show()
        user_win.exec_()

    def _createStatusBar(self):
        self.status = QStatusBar()
        self.status.showMessage("Please enter the required information")
        self.setStatusBar(self.status)

    def progress_event(self, progress):
        self.progress.setValue(progress)

    def phase_handler(self,state):
        if self.sender() == self.phase1_checkBox:
            if state == Qt.Checked:
                self.phase1_ratio_Edit.setEnabled(True)
                self.phase1_text_Edit.setEnabled(True)
            else:
                self.phase1_ratio_Edit.setEnabled(False)
                self.phase1_text_Edit.setEnabled(False)

        if self.sender() == self.phase2_checkBox:
            if state == Qt.Checked:
                self.phase2_ratio_Edit.setEnabled(True)
                self.phase2_text_Edit.setEnabled(True)
            else:
                self.phase2_ratio_Edit.setEnabled(False)
                self.phase2_text_Edit.setEnabled(False)

    def phase_ratio_handler(self):
        if self.sender() == self.phase1_ratio_Edit:
            value = self.phase1_ratio_Edit.value()
            self.phase2_text_Edit.setEnabled(True)
            self.phase2_ratio_Edit.setEnabled(True)
            self.phase2_ratio_Edit.setValue(1-value)
            if value < 1:
                self.phase2_checkBox.setChecked(True)

        if self.sender() == self.phase2_ratio_Edit:
            value = self.phase2_ratio_Edit.value()
            self.phase1_text_Edit.setEnabled(True)
            self.phase1_ratio_Edit.setEnabled(True)
            self.phase1_ratio_Edit.setValue(1-value)
            if value < 1:
                self.phase1_checkBox.setChecked(True)

    def bandwidth_handler(self):
        self.band_width_Edit.setMaximum(self.box_size_Edit.value()/10)

    def button_handler(self):

        if self.sender() == self.phase1_button:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                self.filepath_phase1 = dlg.selectedFiles()
                self.phase1_text_Edit.setText(self.filepath_phase1[0])
        elif self.sender() == self.phase2_button:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                self.filepath_phase2 = dlg.selectedFiles()
                self.phase2_text_Edit.setText(self.filepath_phase2[0])
        elif self.sender() == self.save_button:
            self.save_files = QFileDialog.getExistingDirectory(self)
            self.save_files_Edit.setText(str(self.save_files))


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

    def box_size_handler(self, state):
        if self.sender() == self.box_size_y_check:
            if state == Qt.Checked:
                self.box_size_y_Edit.setEnabled(True)
            else:
                self.box_size_y_Edit.setEnabled(False)

        if self.sender() == self.box_size_z_check:
            if state == Qt.Checked:
                self.box_size_z_Edit.setEnabled(True)
            else:
                self.box_size_z_Edit.setEnabled(False)

            # making other check box to uncheck

    def submit(self):

        phases = []
        box_size = self.box_size_Edit.value()
        if self.box_size_y_check.isChecked():
            box_size_y = self.box_size_y_Edit.value()
        else:
            box_size_y = None

        if self.box_size_z_check.isChecked():
            box_size_z = self.box_size_z_Edit.value()
        else:
            box_size_z = None
        resolution = self.resolution_Edit.value()
        n_rves = self.n_rves_Edit.value()
        n_bands = self.n_bands_Edit.value()
        band_width = self.band_width_Edit.value()

        if not self.phase1_checkBox.isChecked() and not self.phase2_checkBox.isChecked():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please choose at least one phase")
            msg.setInformativeText("Check one of the checkboxes stating\n Ferrite or Martensite")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        phase1_path = None
        phase2_path = None
        phase1_ratio = 0
        phase2_ratio = 0
        if self.phase1_checkBox.isChecked():
            phase1_path = self.phase1_text_Edit.text()
            phase1_ratio = self.phase1_ratio_Edit.value()
            if phase1_path is not None and len(phase1_path) > 0:
                phases.append('ferrite')

        if self.phase2_checkBox.isChecked():
            phase2_path = self.phase2_text_Edit.text()
            phase2_ratio = self.phase2_ratio_Edit.value()
            if phase2_path is not None and len(phase2_path) > 0:
                phases.append('martensite')

        if phase1_ratio == 0:
            phase_ratio = phase2_ratio
        else:
            phase_ratio = phase1_ratio

        if ARGS['subs_flag']:
            if phase2_path is None or len(phase2_path) == 0 or phase2_ratio == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please choose Martnesite or set Martensite \n ratio non-zero!")
                msg.setInformativeText("Check the checkbox stating Martensite \nand set Martensite ratio > 0")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
        store_path = self.save_files_Edit.text()

        store_path_flag = False
        import_flag = False
        dimension = None
        visualization_flag = False
        dimension_flag = False
        gan_flag = False
        equiv_d = ARGS['equiv_d']
        p_sigma = ARGS['p_sigma']
        t_mu = ARGS['t_mu']
        b_sigma = ARGS['b_sigma']
        decreasing_factor = ARGS['decreasing_factor']
        lower = ARGS['lower']
        upper = ARGS['upper']
        circularity = ARGS['circularity']
        save = ARGS['save']
        filename = ARGS['filename']
        subs_file_flag = ARGS['subs_file_flag']
        subs_file = ARGS['subs_file']
        subs_flag = ARGS['subs_flag']
        if phase1_path is None and phase2_path is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No microstructure was imported")
            msg.setInformativeText("Please import a microstructure file of the following type:\n .pkl, .csv")
            msg.setWindowTitle("Error")
            msg.setDetailedText("input data file is missing")
            msg.exec_()
            return

        if phase1_path is not None: #error here
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

        if phase1_path is not None or phase2_path is not None:
            import_flag = True

        if phase2_path is not None and phase1_ratio == 1:
            msg = QMessageBox()
            reply = msg.question(self,'Warning','The second phase file you chose will not be considered in the RVE.\n'
                                 'Are you sure that you want to keep the phase_ratio at 1.0?', msg.Yes | msg.No)
            if reply == msg.Yes:
                pass
            else:
                return

        if phase1_path is not None and phase2_ratio == 1:
            msg = QMessageBox()
            reply = msg.question(self,'Warning','The first phase file you chose will not be considered in the RVE.\n'
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
            self.worker = Worker(box_size=box_size, box_size_y=box_size_y,box_size_z= box_size_z,
                                 resolution=resolution, number_of_rves=n_rves, number_of_bands=n_bands,
                                 bandwidth=band_width, dimension=dimension, visualization_flag=visualization_flag,
                                 file1=phase1_path, file2=phase2_path, phase_ratio=phase_ratio, store_path=store_path,
                                 gui_flag=True, gan_flag=gan_flag,equiv_d=equiv_d,
                                 p_sigma=p_sigma, t_mu=t_mu, b_sigma=b_sigma, decreasing_factor=decreasing_factor,
                                 lower=lower, upper=upper,circularity=circularity, save=save,filename=filename,subs_file_flag=subs_file_flag,subs_file=subs_file,
                                 subs_flag=subs_flag,phases=phases)
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

ckey = False
pskey = False
bskey = False
dkey = False
lkey = False
hkey = False
class UserDefineWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("generate substructure by user-defined values:")
        self.resize(600, 300)
        PacketEquivDLabel = QLabel('packet equiv_d: ')
        self.packet_equiv_d_Edit = QDoubleSpinBox()
        self.packet_equiv_d_Edit.setMinimum(2.0)
        self.packet_equiv_d_Edit.setSingleStep(0.1)

        PacketSigmaLabel = QLabel('sigma(packet size): ')
        self.packet_sigma_Edit = QDoubleSpinBox()
        self.packet_sigma_Edit.setMinimum(0.01)
        self.packet_sigma_Edit.setSingleStep(0.01)
        self.packet_sigma_Check = QCheckBox()
        self.packet_sigma_Edit.setEnabled(False)

        CircularityLabel = QLabel('circularity(packet): ')
        self.circularity_Edit = QDoubleSpinBox()
        self.circularity_Edit.setValue(1.0)
        self.circularity_Edit.setSingleStep(0.1)
        self.circularity_Check = QCheckBox()
        self.circularity_Edit.setEnabled(False)

        def change_circularity_state():
            global ckey
            ckey = not ckey
            self.circularity_Edit.setEnabled(ckey)

        self.circularity_Check.clicked.connect(change_circularity_state)

        def change_PS_state():
            global pskey
            pskey = not pskey
            self.packet_sigma_Edit.setEnabled(pskey)

        self.packet_sigma_Check.clicked.connect(change_PS_state)

        BlockThicknessLabel = QLabel('block thickness: ')
        self.block_thickness_Edit = QDoubleSpinBox()
        self.block_thickness_Edit.setMinimum(0.50)
        self.block_thickness_Edit.setSingleStep(0.1)

        DecreasingFactorLabel = QLabel('decreasing factor: ')
        self.decreasing_factor_Edit = QDoubleSpinBox()
        self.decreasing_factor_Edit.setValue(0.95)
        self.decreasing_factor_Edit.setSingleStep(0.01)
        self.decreasing_factor_Check = QCheckBox()
        self.decreasing_factor_Edit.setEnabled(False)

        def change_D_state():
            global dkey
            dkey = not dkey
            self.decreasing_factor_Edit.setEnabled(dkey)

        self.decreasing_factor_Check.clicked.connect(change_D_state)

        BlockSigmaLabel = QLabel('sigma(block thickness): ')
        self.block_sigma_Edit = QDoubleSpinBox()
        self.block_sigma_Edit.setMinimum(0.01)
        self.block_sigma_Edit.setSingleStep(0.01)
        self.block_sigma_Check = QCheckBox()
        self.block_sigma_Edit.setEnabled(False)

        def change_BS_state():
            global bskey
            bskey = not bskey
            self.block_sigma_Edit.setEnabled(bskey)

        self.block_sigma_Check.clicked.connect(change_BS_state)

        LowerValueLabel = QLabel('lower value(block thickness): ')
        self.lower_value_Edit = QDoubleSpinBox()
        self.lower_value_Edit.setValue(0.0)
        self.lower_value_Edit.setSingleStep(0.1)
        self.lower_value_Check = QCheckBox()
        self.lower_value_Edit.setEnabled(False)

        def change_L_state():
            global lkey
            lkey = not lkey
            self.lower_value_Edit.setEnabled(lkey)

        self.lower_value_Check.clicked.connect(change_L_state)

        HigherValueLabel = QLabel('upper value(block thickness): ')
        self.higher_value_Edit = QDoubleSpinBox()
        self.higher_value_Edit.setValue(1.0)
        self.higher_value_Edit.setSingleStep(0.1)
        self.higher_value_Check = QCheckBox()
        self.higher_value_Edit.setEnabled(False)

        def change_H_state():
            global hkey
            hkey = not hkey
            self.higher_value_Edit.setEnabled(hkey)

        self.higher_value_Check.clicked.connect(change_H_state)

        SaveFileLabel = QLabel('save result(.csv) as: ')
        self.save_file_Edit = QLineEdit()
        self.save_file_Edit.setText('substruct_data.csv')
        self.save_file_Check = QCheckBox()
        self.save_file_Check.setChecked(True)

        self.commitButton = QPushButton()
        self.commitButton.setText('commit')
        self.commitButton.clicked.connect(self.commit)

        grid = QGridLayout()
        grid.setSpacing(15)
        grid.addWidget(PacketEquivDLabel, 1, 0)
        grid.addWidget(self.packet_equiv_d_Edit, 1, 1, alignment=Qt.AlignRight)
        grid.addWidget(PacketSigmaLabel, 2, 0)
        grid.addWidget(self.packet_sigma_Edit, 2, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.packet_sigma_Check, 2, 2, alignment=Qt.AlignRight)
        grid.addWidget(CircularityLabel, 3, 0)
        grid.addWidget(self.circularity_Edit, 3, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.circularity_Check, 3, 2, alignment=Qt.AlignRight)
        grid.addWidget(BlockThicknessLabel, 4, 0)
        grid.addWidget(self.block_thickness_Edit, 4, 1, alignment=Qt.AlignRight)
        grid.addWidget(DecreasingFactorLabel, 5, 0)
        grid.addWidget(self.decreasing_factor_Edit, 5, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.decreasing_factor_Check, 5, 2, alignment=Qt.AlignRight)
        grid.addWidget(BlockSigmaLabel, 6, 0)
        grid.addWidget(self.block_sigma_Edit, 6, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.block_sigma_Check, 6, 2, alignment=Qt.AlignRight)
        grid.addWidget(LowerValueLabel, 7, 0)
        grid.addWidget(self.lower_value_Edit, 7, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.lower_value_Check, 7, 2, alignment=Qt.AlignRight)
        grid.addWidget(HigherValueLabel, 8, 0)
        grid.addWidget(self.higher_value_Edit, 8, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.higher_value_Check, 8, 2, alignment=Qt.AlignRight)
        grid.addWidget(SaveFileLabel, 9, 0)
        grid.addWidget(self.save_file_Edit,9, 1)
        grid.addWidget(self.save_file_Check, 9, 2, alignment=Qt.AlignRight)
        grid.addWidget(self.commitButton, 10, 2,alignment=Qt.AlignRight)

        self.setLayout(grid)

    def commit(self):
        ARGS['subs_flag'] = True
        ARGS['equiv_d'] = self.packet_equiv_d_Edit.value()
        ARGS['t_mu'] = self.block_thickness_Edit.value()
        ARGS['subs_file_flag'] = False
        ARGS['decreasing_factor'] = 0.95
        if self.decreasing_factor_Check.isChecked():
            ARGS['decreasing_factor'] = self.decreasing_factor_Edit.value()
        ARGS['circularity'] = 1.0
        if self.circularity_Check.isChecked():
            ARGS['circularity'] = self.circularity_Edit.value()
        ARGS['p_sigma'] = 0.01
        if self.packet_sigma_Check.isChecked():
            ARGS['p_sigma'] = self.packet_sigma_Edit.value()
        ARGS['b_sigma'] = 0.01
        if self.block_sigma_Check.isChecked():
            ARGS['b_sigma'] = self.block_sigma_Edit.value()
        ARGS['lower'] = None
        if self.lower_value_Check.isChecked():
            ARGS['lower'] = self.lower_value_Edit.value()
        ARGS['upper'] = None
        if self.higher_value_Check.isChecked():
            ARGS['upper'] = self.higher_value_Edit.value()
        ARGS['save'] = True
        if self.save_file_Check.isChecked():
            ARGS['save'] = True
            ARGS['filename'] = self.save_file_Edit.text()
        else:
            ARGS['save'] = False

        ARGS['subs_file'] = None

        for key,item in ARGS.items():
            print(key,': ',item)

        self.close()

class FromFileWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(600, 300)
        self.setWindowTitle('Generate from files: ')
        SubsFileLabel = QLabel('substructure file path: ')
        self.subs_file_Edit = QLineEdit()
        self.subs_file_button = QPushButton()
        self.subs_file_button.setText('select_file')
        self.subs_file_button.clicked.connect(self.getfiles)

        PacketSigmaLabel = QLabel('sigma(packet size): ')
        self.packet_sigma_Edit = QDoubleSpinBox()
        self.packet_sigma_Edit.setMinimum(0.01)
        self.packet_sigma_Edit.setSingleStep(0.01)
        self.packet_sigma_Check = QCheckBox()
        self.packet_sigma_Edit.setEnabled(False)
        def change_PS_state():
            global pskey
            pskey = not pskey
            self.packet_sigma_Edit.setEnabled(pskey)
        self.packet_sigma_Check.clicked.connect(change_PS_state)

        BlockSigmaLabel = QLabel('sigma(block thickness): ')
        self.block_sigma_Edit = QDoubleSpinBox()
        self.block_sigma_Edit.setMinimum(0.01)
        self.block_sigma_Edit.setSingleStep(0.01)
        self.block_sigma_Check = QCheckBox()
        self.block_sigma_Edit.setEnabled(False)
        def change_BS_state():
            global bskey
            bskey = not bskey
            self.block_sigma_Edit.setEnabled(bskey)
        self.block_sigma_Check.clicked.connect(change_BS_state)

        DecreasingFactorLabel = QLabel('decreasing factor: ')
        self.decreasing_factor_Edit = QDoubleSpinBox()
        self.decreasing_factor_Edit.setValue(0.95)
        self.decreasing_factor_Edit.setSingleStep(0.01)
        self.decreasing_factor_Check = QCheckBox()
        self.decreasing_factor_Edit.setEnabled(False)
        def change_D_state():
            global dkey
            dkey = not dkey
            self.decreasing_factor_Edit.setEnabled(dkey)
        self.decreasing_factor_Check.clicked.connect(change_D_state)

        LowerValueLabel = QLabel('lower value(block thickness): ')
        self.lower_value_Edit = QDoubleSpinBox()
        self.lower_value_Edit.setValue(0.0)
        self.lower_value_Edit.setSingleStep(0.1)
        self.lower_value_Check = QCheckBox()
        self.lower_value_Edit.setEnabled(False)
        def change_L_state():
            global lkey
            lkey = not lkey
            self.lower_value_Edit.setEnabled(lkey)
        self.lower_value_Check.clicked.connect(change_L_state)

        HigherValueLabel = QLabel('upper value(block thickness): ')
        self.higher_value_Edit = QDoubleSpinBox()
        self.higher_value_Edit.setValue(1.0)
        self.higher_value_Edit.setSingleStep(0.1)
        self.higher_value_Check = QCheckBox()
        self.higher_value_Edit.setEnabled(False)
        def change_H_state():
            global hkey
            hkey = not hkey
            self.lower_value_Edit.setEnabled(hkey)
        self.higher_value_Check.clicked.connect(change_H_state)

        SaveFileLabel = QLabel('save result(.csv) as: ')
        self.save_file_Edit = QLineEdit()
        self.save_file_Edit.setText('substruct_data.csv')
        self.save_file_Check = QCheckBox()
        self.save_file_Check.setChecked(True)

        self.commitButton = QPushButton()
        self.commitButton.setText('commit')
        self.commitButton.clicked.connect(self.commit)

        grid = QGridLayout()
        grid.setSpacing(15)
        grid.addWidget(SubsFileLabel, 1, 0)
        grid.addWidget(self.subs_file_Edit, 1, 1, 1, 2)
        grid.addWidget(self.subs_file_button, 1, 3)
        grid.addWidget(PacketSigmaLabel, 2, 0)
        grid.addWidget(self.packet_sigma_Edit, 2, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.packet_sigma_Check, 2, 2, alignment=Qt.AlignRight)
        grid.addWidget(BlockSigmaLabel, 3, 0)
        grid.addWidget(self.block_sigma_Edit, 3, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.block_sigma_Check, 3, 2, alignment=Qt.AlignRight)
        grid.addWidget(DecreasingFactorLabel, 4, 0)
        grid.addWidget(self.decreasing_factor_Edit, 4, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.decreasing_factor_Check, 4, 2, alignment=Qt.AlignRight)
        grid.addWidget(LowerValueLabel,5,0)
        grid.addWidget(self.lower_value_Edit, 5, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.lower_value_Check, 5, 2, alignment=Qt.AlignRight)
        grid.addWidget(HigherValueLabel, 6, 0)
        grid.addWidget(self.higher_value_Edit, 6, 1, alignment=Qt.AlignRight)
        grid.addWidget(self.higher_value_Check, 6, 2, alignment=Qt.AlignRight)
        grid.addWidget(SaveFileLabel,7,0)
        grid.addWidget(self.save_file_Edit,7,1)
        grid.addWidget(self.save_file_Check,7,2,alignment=Qt.AlignRight)
        grid.addWidget(self.commitButton,8,3,alignment=Qt.AlignRight)

        self.setLayout(grid)

    def commit(self):
        ARGS['subs_file_flag'] = True
        ARGS['subs_flag'] = True
        ARGS['subs_file'] = self.subs_file_Edit.text()
        ARGS['equiv_d'] = None

        ARGS['t_mu'] = None
        ARGS['circularity'] = 1.0
        ARGS['decreasing_factor'] = 0.95
        if self.decreasing_factor_Check.isChecked():
            ARGS['decreasing_factor'] = self.decreasing_factor_Edit.value()
        ARGS['p_sigma'] = 0.01
        if self.packet_sigma_Check.isChecked():
            ARGS['p_sigma'] = self.packet_sigma_Edit.value()
        ARGS['b_sigma'] = 0.01
        if self.block_sigma_Check.isChecked():
            ARGS['b_sigma'] = self.block_sigma_Edit.value()
        ARGS['lower'] = None
        if self.lower_value_Check.isChecked():
            ARGS['lower'] = self.lower_value_Edit.value()
        ARGS['upper'] = None
        if self.higher_value_Check.isChecked():
            ARGS['upper'] = self.higher_value_Edit.value()
        ARGS['save'] = True
        if self.save_file_Check.isChecked():
            ARGS['save'] = True
            ARGS['filename'] = self.save_file_Edit.text()
        else:
            ARGS['save'] = False
        for key, item in ARGS.items():
            print(key,': ',item)
        self.close()

    def getfiles(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            self.subs_files = dlg.selectedFiles()
            self.subs_file_Edit.setText(self.subs_files[0])

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
    win.resize(1000,800)
    win.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QIcon
from dragen.pyqt_gui.worker import Worker


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./dragen/thumbnails/Folder-Generic-Silver-icon.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.MainWindow = MainWindow
        MainWindow.setObjectName("Main Window")
        MainWindow.setWindowTitle("DRAGen - RVE Generator")
        MainWindow.setFixedSize(930, 850) #change
        self.thumbnail_path = sys.argv[0][:-10] + "\\dragen\\thumbnails\\"
        MainWindow.setWindowIcon(QIcon(self.thumbnail_path + '\\Drache.ico'))

        # Definition of Containers:
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 460, 910, 330)) #change
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(10, 460, 910, 330)) #change
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.InfoPages = QtWidgets.QTabWidget(self.scrollAreaWidgetContents)
        self.InfoPages.setEnabled(True)
        self.InfoPages.setGeometry(QtCore.QRect(-5, 0, 910, 330)) #change
        self.InfoPages.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.InfoPages.setUsesScrollButtons(True)
        self.InfoPages.setTabsClosable(False)
        self.InfoPages.setMovable(False)
        self.InfoPages.setTabBarAutoHide(False)
        self.InfoPages.setObjectName("InfoPages")





        ### Tab window
        # Status Tab

        self.status_tab = QtWidgets.QWidget()
        self.InfoPages.addTab(self.status_tab, "")
        self.status_tab.setObjectName("status_tab")

        self.textBrowser = QtWidgets.QTextBrowser(self.status_tab)
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 981, 336)) #change
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setText(" Please enter the required information")


        self.verticalScrollBar = QtWidgets.QScrollBar(self.status_tab)
        self.verticalScrollBar.setGeometry(QtCore.QRect(980, 0, 20, 336)) #change
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")

        # Banding Feature Tab
        self.banding_tab = QtWidgets.QWidget()
        self.banding_tab.setObjectName("banding_tab")
        self.InfoPages.addTab(self.banding_tab, "")
        self.banding_tab.setEnabled(False)

        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.banding_tab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(20, 10, 500, 180))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.NoBand_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.NoBand_label.sizePolicy().hasHeightForWidth())
        self.NoBand_label.setSizePolicy(sizePolicy)
        self.NoBand_label.setObjectName("NoBand_label")
        self.gridLayout_3.addWidget(self.NoBand_label, 0, 0, 1, 2)

        self.NoBandsSpinBox = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        self.NoBandsSpinBox.setMinimum(1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.NoBandsSpinBox.sizePolicy().hasHeightForWidth())
        self.NoBandsSpinBox.setSizePolicy(sizePolicy)
        self.NoBandsSpinBox.setObjectName("NoBandsSpinBox")
        self.gridLayout_3.addWidget(self.NoBandsSpinBox, 0, 2, 1, 1)

        self.band_lower_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_lower_label.sizePolicy().hasHeightForWidth())
        self.band_lower_label.setSizePolicy(sizePolicy)
        self.band_lower_label.setObjectName("band_lower_label")
        self.gridLayout_3.addWidget(self.band_lower_label, 1, 0, 1, 2)

        self.band_lowerSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_lowerSpinBox.sizePolicy().hasHeightForWidth())
        self.band_lowerSpinBox.setSizePolicy(sizePolicy)
        self.band_lowerSpinBox.setObjectName("band_lowerSpinBox")
        self.gridLayout_3.addWidget(self.band_lowerSpinBox, 1, 2, 1, 1)
        self.band_lowerSpinBox.valueChanged.connect(self.bandwidth_handler)


        self.band_upper_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_upper_label.sizePolicy().hasHeightForWidth())
        self.band_upper_label.setSizePolicy(sizePolicy)
        self.band_upper_label.setObjectName("band_upper_label")
        self.gridLayout_3.addWidget(self.band_upper_label, 2, 0, 1, 2)

        self.band_upperSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_upperSpinBox.sizePolicy().hasHeightForWidth())
        self.band_upperSpinBox.setSizePolicy(sizePolicy)
        self.band_upperSpinBox.setObjectName("band_upperSpinBox")
        self.gridLayout_3.addWidget(self.band_upperSpinBox, 2, 2, 1, 1)
        self.band_upperSpinBox.valueChanged.connect(self.bandwidth_handler)

        self.band_filling_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_filling_label.sizePolicy().hasHeightForWidth())
        self.band_filling_label.setSizePolicy(sizePolicy)
        self.band_filling_label.setObjectName("band_filling_label")
        self.gridLayout_3.addWidget(self.band_filling_label, 3, 0, 1, 2)

        self.band_fillingSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_fillingSpinBox.sizePolicy().hasHeightForWidth())
        self.band_fillingSpinBox.setMinimum(0.1)
        self.band_fillingSpinBox.setMaximum(0.7)
        self.band_fillingSpinBox.setSizePolicy(sizePolicy)
        self.band_fillingSpinBox.setObjectName("band_fillingSpinBox")
        self.gridLayout_3.addWidget(self.band_fillingSpinBox, 3, 2, 1, 1)
        self.band_fillingSpinBox.valueChanged.connect(self.bandwidth_handler)

        self.band_orientation_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_orientation_label.sizePolicy().hasHeightForWidth())

        self.band_orientation_label.setSizePolicy(sizePolicy)
        self.band_orientation_label.setObjectName("band_orientation_label")
        self.gridLayout_3.addWidget(self.band_orientation_label, 4, 0, 1, 1)

        self.BandOrientation_XY = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.BandOrientation_XY.setObjectName("BandOrientation_XY")
        self.gridLayout_3.addWidget(self.BandOrientation_XY, 4, 2, 1, 1)
        self.BandOrientation_XY.setChecked(True)
        self.BandOrientation_XY.setText('xy')

        self.BandOrientation_XZ = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.BandOrientation_XZ.setObjectName("BandOrientation_XZ")
        self.gridLayout_3.addWidget(self.BandOrientation_XZ, 4, 3, 1, 1)
        self.BandOrientation_XZ.setText('xz')

        self.BandOrientation_YZ = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.BandOrientation_YZ.setObjectName("BandOrientation_YZ")
        self.gridLayout_3.addWidget(self.BandOrientation_YZ, 4, 4, 1, 1)
        self.BandOrientation_YZ.setText('yz')


        self.band_file_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_file_label.sizePolicy().hasHeightForWidth())

        self.band_file_label.setSizePolicy(sizePolicy)
        self.band_file_label.setObjectName("band_file_label")
        self.gridLayout_3.addWidget(self.band_file_label, 5, 0, 1, 1)

        self.lineEditBand = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.lineEditBand.setObjectName("lineEditBand")
        self.gridLayout_3.addWidget(self.lineEditBand, 5, 2, 1, 3)

        self.fileBrowserBand = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserBand.sizePolicy().hasHeightForWidth())
        self.fileBrowserBand.setSizePolicy(sizePolicy)
        self.fileBrowserBand.setText("")
        self.fileBrowserBand.setIcon(icon)
        self.fileBrowserBand.setObjectName("fileBrowserBand")
        self.gridLayout_3.addWidget(self.fileBrowserBand, 5, 5, 1, 1)
        self.fileBrowserBand.clicked.connect(self.button_handler)

        # Inclusion Feature Tab
        self.inclusion_tab = QtWidgets.QWidget()
        self.inclusion_tab.setObjectName("inclusion_tab")
        self.InfoPages.addTab(self.inclusion_tab, "")
        self.inclusion_tab.setEnabled(False)

        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.inclusion_tab)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(20, 10, 230, 120))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")

        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")

        self.inclusion_file_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inclusion_file_label.sizePolicy().hasHeightForWidth())

        self.inclusion_file_label.setSizePolicy(sizePolicy)
        self.inclusion_file_label.setObjectName("inclusion_file_label")
        self.gridLayout_4.addWidget(self.inclusion_file_label, 0, 0, 1, 2)

        self.lineEditInclusion = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.lineEditInclusion.setObjectName("lineEditInclusion")
        self.gridLayout_4.addWidget(self.lineEditInclusion, 0, 2, 1, 1)

        self.fileBrowserInclusion = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserInclusion.sizePolicy().hasHeightForWidth())
        self.fileBrowserInclusion.setSizePolicy(sizePolicy)
        self.fileBrowserInclusion.setText("")
        self.fileBrowserInclusion.setIcon(icon)
        self.fileBrowserInclusion.setObjectName("fileBrowserInclusion")
        self.gridLayout_4.addWidget(self.fileBrowserInclusion, 0, 3, 1, 1)
        self.fileBrowserInclusion.clicked.connect(self.button_handler)

        self.inclusion_ratio_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inclusion_ratio_label.sizePolicy().hasHeightForWidth())

        self.inclusion_ratio_label.setSizePolicy(sizePolicy)
        self.inclusion_ratio_label.setObjectName("inclusion_ratio_label")
        self.gridLayout_4.addWidget(self.inclusion_ratio_label, 1, 0, 1, 2)

        self.inclusionSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_3)
        self.inclusionSpinBox.setMinimum(0.01)
        self.lineEditInclusion.setObjectName("inclusionSpinBox")
        self.gridLayout_4.addWidget(self.inclusionSpinBox, 1, 2, 1, 1)

        # Substructure Feature Tab
        self.substructure = QtWidgets.QWidget()
        self.substructure.setObjectName("substructure")
        self.InfoPages.addTab(self.substructure, "")
        self.substructure.setEnabled(False)

        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.substructure)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 0, 240, 21))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")

        self.grid_mode = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.grid_mode.setContentsMargins(0, 0, 0, 0)
        self.grid_mode.setObjectName("grid_mode")

        self.substructure_filemode_radio = QtWidgets.QRadioButton(self.gridLayoutWidget_4)
        self.substructure_filemode_radio.setChecked(True)
        self.substructure_filemode_radio.setObjectName("substructure_filemode_radio")
        self.grid_mode.addWidget(self.substructure_filemode_radio, 0, 0, 1, 2)
        self.substructure_filemode_radio.toggled.connect(self.widget_handler)

        self.substructure_user_mode_radio = QtWidgets.QRadioButton(self.gridLayoutWidget_4)
        self.substructure_user_mode_radio.setChecked(False)
        self.substructure_user_mode_radio.setObjectName("substructure_user_mode_radio")
        self.grid_mode.addWidget(self.substructure_user_mode_radio, 0, 2, 1, 2)
        self.substructure_user_mode_radio.toggled.connect(self.widget_handler)

        self.tabWidget = QtWidgets.QTabWidget(self.substructure)
        self.tabWidget.setGeometry(QtCore.QRect(6, 29, 721, 461))
        self.tabWidget.setObjectName("tabWidget")

        # Substructure Feature Tab -> File Tab
        self.file_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(self.file_tab, "")
        self.file_tab.setObjectName("file_tab")

        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.file_tab)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(0, 0, 421, 201))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")

        self.grid_file = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.grid_file.setContentsMargins(0, 0, 0, 0)
        self.grid_file.setObjectName("grid_file")

        self.label_substructure_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_substructure_file.setObjectName("label_substructure_file")
        self.grid_file.addWidget(self.label_substructure_file, 0, 0, 1, 1)

        self.substructure_file_lineEdit_file = QtWidgets.QLineEdit(self.gridLayoutWidget_5)
        self.substructure_file_lineEdit_file.setObjectName("substructure_file_lineEdit_file")
        self.grid_file.addWidget(self.substructure_file_lineEdit_file, 0, 1, 1, 1)

        self.substructure_file_browser_file = QtWidgets.QPushButton(self.gridLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.substructure_file_browser_file.sizePolicy().hasHeightForWidth())
        self.substructure_file_browser_file.setSizePolicy(sizePolicy)
        self.substructure_file_browser_file.setText("")

        self.substructure_file_browser_file.setIcon(icon)
        self.substructure_file_browser_file.setObjectName("substructure_file_browser_file")
        self.grid_file.addWidget(self.substructure_file_browser_file, 0, 2, 1, 1)
        self.substructure_file_browser_file.clicked.connect(self.button_handler)

        self.label_packet_size_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_packet_size_file.setObjectName("label_packet_size_file")
        self.grid_file.addWidget(self.label_packet_size_file, 1, 0, 1, 1)

        self.substructure_packet_size_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_packet_size_SpinBox_file.setEnabled(False)
        self.substructure_packet_size_SpinBox_file.setMinimum(0.01)
        self.substructure_packet_size_SpinBox_file.setSingleStep(0.01)
        self.substructure_packet_size_SpinBox_file.setObjectName("substructure_packet_size_SpinBox")
        self.grid_file.addWidget(self.substructure_packet_size_SpinBox_file, 1, 1, 1, 1)

        self.substructure_packet_size_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_packet_size_checkBox_file.setText("")
        self.substructure_packet_size_checkBox_file.setObjectName("substructure_packet_size_checkBox_file")
        self.grid_file.addWidget(self.substructure_packet_size_checkBox_file, 1, 2, 1, 1)
        self.substructure_packet_size_checkBox_file.stateChanged.connect(self.substructure_file_checkbox_handler)

        self.label_block_thicknes_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_block_thicknes_file.setObjectName("label_block_thicknes_file")
        self.grid_file.addWidget(self.label_block_thicknes_file, 2, 0, 1, 1)

        self.substructure_block_thickness_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_block_thickness_SpinBox_file.setEnabled(False)
        self.substructure_block_thickness_SpinBox_file.setMinimum(0.01)
        self.substructure_block_thickness_SpinBox_file.setSingleStep(0.01)
        self.substructure_block_thickness_SpinBox_file.setObjectName("substructure_block_thickness_SpinBox_file")
        self.grid_file.addWidget(self.substructure_block_thickness_SpinBox_file, 2, 1, 1, 1)

        self.substructure_block_thickness_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_block_thickness_checkBox_file.setText("")
        self.substructure_block_thickness_checkBox_file.setObjectName("substructure_block_thickness_checkBox_file")
        self.grid_file.addWidget(self.substructure_block_thickness_checkBox_file, 2, 2, 1, 1)
        self.substructure_block_thickness_checkBox_file.stateChanged.connect(self.substructure_file_checkbox_handler)

        self.label_decreasing_fact_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_decreasing_fact_file.setObjectName("label_decreasing_fact_file")
        self.grid_file.addWidget(self.label_decreasing_fact_file, 3, 0, 1, 1)

        self.substructure_decreasing_fact_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_decreasing_fact_SpinBox_file.setEnabled(False)
        self.substructure_decreasing_fact_SpinBox_file.setValue(0.95)
        self.substructure_decreasing_fact_SpinBox_file.setSingleStep(0.01)
        self.substructure_decreasing_fact_SpinBox_file.setObjectName("substructure_decreasing_fact_SpinBox_file")
        self.grid_file.addWidget(self.substructure_decreasing_fact_SpinBox_file, 3, 1, 1, 1)

        self.substructure_decreasing_fact_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_decreasing_fact_checkBox_file.setText("")
        self.substructure_decreasing_fact_checkBox_file.setObjectName("substructure_decreasing_fact_checkBox_file")
        self.grid_file.addWidget(self.substructure_decreasing_fact_checkBox_file, 3, 2, 1, 1)
        self.substructure_decreasing_fact_checkBox_file.stateChanged.connect(self.substructure_file_checkbox_handler)

        self.label_min_block_thickness_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_min_block_thickness_file.setObjectName("label_min_block_thickness_file")
        self.grid_file.addWidget(self.label_min_block_thickness_file, 4, 0, 1, 1)

        self.substructure_min_block_thickness_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_min_block_thickness_SpinBox_file.setEnabled(False)
        self.substructure_min_block_thickness_SpinBox_file.setValue(0.0)
        self.substructure_min_block_thickness_SpinBox_file.setSingleStep(0.1)
        self.substructure_min_block_thickness_SpinBox_file.setObjectName("substructure_min_block_thickness_SpinBox_file")
        self.grid_file.addWidget(self.substructure_min_block_thickness_SpinBox_file, 4, 1, 1, 1)

        self.substructure_min_block_thickness_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_min_block_thickness_checkBox_file.setText("")
        self.substructure_min_block_thickness_checkBox_file.setObjectName("substructure_min_block_thickness_checkBox_file")
        self.grid_file.addWidget(self.substructure_min_block_thickness_checkBox_file, 4, 2, 1, 1)
        self.substructure_min_block_thickness_checkBox_file.stateChanged.connect(self.substructure_file_checkbox_handler)

        self.label_max_block_thickness_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_max_block_thickness_file.setObjectName("label_max_block_thickness_file")
        self.grid_file.addWidget(self.label_max_block_thickness_file, 5, 0, 1, 1)

        self.substructure_max_block_thickness_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_max_block_thickness_SpinBox_file.setEnabled(False)
        self.substructure_max_block_thickness_SpinBox_file.setValue(1.0)
        self.substructure_max_block_thickness_SpinBox_file.setSingleStep(0.1)
        self.substructure_max_block_thickness_SpinBox_file.setObjectName("substructure_max_block_thickness_SpinBox_file")
        self.grid_file.addWidget(self.substructure_max_block_thickness_SpinBox_file, 5, 1, 1, 1)

        self.substructure_max_block_thickness_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_max_block_thickness_checkBox_file.setText("")
        self.substructure_max_block_thickness_checkBox_file.setObjectName("substructure_max_block_thickness_checkBox_file")
        self.grid_file.addWidget(self.substructure_max_block_thickness_checkBox_file, 5, 2, 1, 1)
        self.substructure_max_block_thickness_checkBox_file.stateChanged.connect(self.substructure_file_checkbox_handler)

        self.label_save_result_file = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_save_result_file.setObjectName("label_save_result_file")
        self.grid_file.addWidget(self.label_save_result_file, 6, 0, 1, 1)

        self.substructure_save_result_lineEdit_file = QtWidgets.QLineEdit(self.gridLayoutWidget_5)
        self.substructure_save_result_lineEdit_file.setEnabled(False)
        self.substructure_save_result_lineEdit_file.setText('substructure_data.csv')
        self.substructure_save_result_lineEdit_file.setObjectName("substructure_save_result_lineEdit_file")
        self.grid_file.addWidget(self.substructure_save_result_lineEdit_file, 6, 1, 1, 1)

        self.substructure_save_result_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_save_result_checkBox_file.setText("")
        self.substructure_save_result_checkBox_file.setObjectName("substructure_save_result_checkBox_file")
        self.grid_file.addWidget(self.substructure_save_result_checkBox_file, 6, 2, 1, 1)
        self.substructure_save_result_checkBox_file.stateChanged.connect(self.substructure_file_checkbox_handler)

        # Substructure Feature Tab -> User Tab
        self.user_tab = QtWidgets.QWidget()
        self.user_tab.setObjectName("user_tab")
        self.tabWidget.addTab(self.user_tab, "")

        self.gridLayoutWidget_6 = QtWidgets.QWidget(self.user_tab)
        self.gridLayoutWidget_6.setGeometry(QtCore.QRect(0, 3, 901, 238)) #changetry
        self.gridLayoutWidget_6.setObjectName("gridLayoutWidget_6")

        self.grid_user = QtWidgets.QGridLayout(self.gridLayoutWidget_6)
        self.grid_user.setContentsMargins(0, 0, 0, 0)
        self.grid_user.setObjectName("grid_user")

        self.label_packet_eq_d_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_packet_eq_d_user.setObjectName("label_packet_eq_d_user")
        self.grid_user.addWidget(self.label_packet_eq_d_user, 0, 0, 1, 1)

        self.substructure_packet_eq_d_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_packet_eq_d_SpinBox_user.setEnabled(True)
        self.substructure_packet_eq_d_SpinBox_user.setMinimum(2.0)
        self.substructure_packet_eq_d_SpinBox_user.setSingleStep(0.1)
        self.substructure_packet_eq_d_SpinBox_user.setObjectName("substructure_packet_eq_d_SpinBox_user")
        self.grid_user.addWidget(self.substructure_packet_eq_d_SpinBox_user, 0, 1, 1, 1)

        self.label_packet_size_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_packet_size_user.setObjectName("label_packet_size_user")
        self.grid_user.addWidget(self.label_packet_size_user, 1, 0, 1, 1)

        self.substructure_packet_size_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_packet_size_SpinBox_user.setEnabled(False)
        self.substructure_packet_size_SpinBox_user.setMinimum(0.01)
        self.substructure_packet_size_SpinBox_user.setSingleStep(0.01)
        self.substructure_packet_size_SpinBox_user.setObjectName("substructure_packet_size_SpinBox_user")
        self.grid_user.addWidget(self.substructure_packet_size_SpinBox_user, 1, 1, 1, 1)

        self.substructure_packet_size_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_packet_size_checkBox_user.setText("")
        self.substructure_packet_size_checkBox_user.setObjectName("substructure_packet_size_checkBox_user")
        self.grid_user.addWidget(self.substructure_packet_size_checkBox_user, 1, 2, 1, 1)
        self.substructure_packet_size_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        self.label_circularity_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_circularity_user.setObjectName("label_circularity_user")
        self.grid_user.addWidget(self.label_circularity_user, 2, 0, 1, 1)

        self.substructure_circularity_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_circularity_SpinBox_user.setEnabled(False)
        self.substructure_circularity_SpinBox_user.setValue(1.0)
        self.substructure_circularity_SpinBox_user.setSingleStep(0.1)
        self.substructure_circularity_SpinBox_user.setObjectName("substructure_circularity_SpinBox_user")
        self.grid_user.addWidget(self.substructure_circularity_SpinBox_user, 2, 1, 1, 1)

        self.substructure_circularity_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_circularity_checkBox_user.setText("")
        self.substructure_circularity_checkBox_user.setObjectName("substructure_circularity_checkBox_user")
        self.grid_user.addWidget(self.substructure_circularity_checkBox_user, 2, 2, 1, 1)
        self.substructure_circularity_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        self.label_block_thickness_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_block_thickness_user.setObjectName("label_block_thickness_user")
        self.grid_user.addWidget(self.label_block_thickness_user, 3, 0, 1, 1)

        self.substructure_block_thickness_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_block_thickness_SpinBox_user.setEnabled(True)
        self.substructure_block_thickness_SpinBox_user.setMinimum(0.50)
        self.substructure_block_thickness_SpinBox_user.setSingleStep(0.1)
        self.substructure_block_thickness_SpinBox_user.setObjectName("substructure_block_thickness_SpinBox_user")
        self.grid_user.addWidget(self.substructure_block_thickness_SpinBox_user, 3, 1, 1, 1)

        self.label_decreasing_fact_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_decreasing_fact_user.setObjectName("label_decreasing_fact_user")
        self.grid_user.addWidget(self.label_decreasing_fact_user, 4, 0, 1, 1)

        self.substructure_decreasing_fact_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_decreasing_fact_SpinBox_user.setEnabled(False)
        self.substructure_decreasing_fact_SpinBox_user.setValue(0.95)
        self.substructure_decreasing_fact_SpinBox_user.setSingleStep(0.01)
        self.substructure_decreasing_fact_SpinBox_user.setObjectName("substructure_decreasing_fact_SpinBox_user")
        self.grid_user.addWidget(self.substructure_decreasing_fact_SpinBox_user, 4, 1, 1, 1)

        self.substructure_decreasing_fact_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_decreasing_fact_checkBox_user.setText("")
        self.substructure_decreasing_fact_checkBox_user.setObjectName("substructure_decreasing_fact_checkBox_user")
        self.grid_user.addWidget(self.substructure_decreasing_fact_checkBox_user, 4, 2, 1, 1)
        self.substructure_decreasing_fact_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        self.label_block_thickness_sigma_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_block_thickness_sigma_user.setObjectName("label_block_thickness_sigma_user")
        self.grid_user.addWidget(self.label_block_thickness_sigma_user, 5, 0, 1, 1)

        self.substructure_block_thickness_sigma_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_block_thickness_sigma_SpinBox_user.setEnabled(False)
        self.substructure_block_thickness_sigma_SpinBox_user.setMinimum(0.01)
        self.substructure_block_thickness_sigma_SpinBox_user.setSingleStep(0.01)
        self.substructure_block_thickness_sigma_SpinBox_user.setObjectName("substructure_block_thickness_sigma_SpinBox_user")
        self.grid_user.addWidget(self.substructure_block_thickness_sigma_SpinBox_user, 5, 1, 1, 1)

        self.substructure_block_thickness_sigma_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_block_thickness_sigma_checkBox_user.setText("")
        self.substructure_block_thickness_sigma_checkBox_user.setObjectName("substructure_block_thickness_sigma_checkBox_user")
        self.grid_user.addWidget(self.substructure_block_thickness_sigma_checkBox_user, 5, 2, 1, 1)
        self.substructure_block_thickness_sigma_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        self.label_min_block_thickness_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_min_block_thickness_user.setObjectName("label_min_block_thickness_user")
        self.grid_user.addWidget(self.label_min_block_thickness_user, 6, 0, 1, 1)

        self.substructure_min_block_thickness_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_min_block_thickness_SpinBox_user.setEnabled(False)
        self.substructure_min_block_thickness_SpinBox_user.setValue(0.0)
        self.substructure_min_block_thickness_SpinBox_user.setSingleStep(0.1)
        self.substructure_min_block_thickness_SpinBox_user.setObjectName("substructure_min_block_thickness_SpinBox_user")
        self.grid_user.addWidget(self.substructure_min_block_thickness_SpinBox_user, 6, 1, 1, 1)

        self.substructure_min_block_thickness_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_min_block_thickness_checkBox_user.setText("")
        self.substructure_min_block_thickness_checkBox_user.setObjectName("substructure_min_block_thickness_checkBox_user")
        self.grid_user.addWidget(self.substructure_min_block_thickness_checkBox_user, 6, 2, 1, 1)
        self.substructure_min_block_thickness_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        self.label_max_block_thickness_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_max_block_thickness_user.setObjectName("label_max_block_thickness_user")
        self.grid_user.addWidget(self.label_max_block_thickness_user, 7, 0, 1, 1)

        self.substructure_max_block_thickness_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_6)
        self.substructure_max_block_thickness_SpinBox_user.setEnabled(False)
        self.substructure_max_block_thickness_SpinBox_user.setValue(1.0)
        self.substructure_max_block_thickness_SpinBox_user.setSingleStep(0.1)
        self.substructure_max_block_thickness_SpinBox_user.setObjectName("substructure_max_block_thickness_SpinBox_user")
        self.grid_user.addWidget(self.substructure_max_block_thickness_SpinBox_user, 7, 1, 1, 1)

        self.substructure_max_block_thickness_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_max_block_thickness_checkBox_user.setText("")
        self.substructure_max_block_thickness_checkBox_user.setObjectName("substructure_max_block_thickness_checkBox_user")
        self.grid_user.addWidget(self.substructure_max_block_thickness_checkBox_user, 7, 2, 1, 1)
        self.substructure_max_block_thickness_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        self.label_save_result_user = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.label_save_result_user.setObjectName("label_save_result_user")
        self.grid_user.addWidget(self.label_save_result_user, 8, 0, 1, 1)

        self.substructure_save_result_lineEdit_user = QtWidgets.QLineEdit(self.gridLayoutWidget_6)
        self.substructure_save_result_lineEdit_user.setEnabled(True)
        self.substructure_save_result_lineEdit_user.setText('substructure_data.csv')
        self.substructure_save_result_lineEdit_user.setObjectName("substructure_save_result_lineEdit_user")
        self.grid_user.addWidget(self.substructure_save_result_lineEdit_user, 8, 1, 1, 1)

        self.substructure_save_result_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_6)
        self.substructure_save_result_checkBox_user.setText("")
        self.substructure_save_result_checkBox_user.setChecked(True)
        self.substructure_save_result_checkBox_user.setObjectName("substructure_save_result_checkBox_user")
        self.grid_user.addWidget(self.substructure_save_result_checkBox_user, 8, 2, 1, 1)
        self.substructure_save_result_checkBox_user.stateChanged.connect(self.substructure_user_checkbox_handler)

        # Visualization Tab
        self.visualization_tab = QtWidgets.QWidget()
        self.visualization_tab.setObjectName("visualization_tab")
        self.InfoPages.addTab(self.visualization_tab, "")

        self.gridLayoutWidget_7 = QtWidgets.QWidget(self.visualization_tab)
        self.gridLayoutWidget_7.setGeometry(QtCore.QRect(20, 10, 230, 120))
        self.gridLayoutWidget_7.setObjectName("gridLayoutWidget_7")

        self.gridLayout_6 = QtWidgets.QGridLayout(self.gridLayoutWidget_7)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")

        self.visualization_flag_button = QtWidgets.QCheckBox(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.visualization_flag_button.sizePolicy().hasHeightForWidth())
        self.gridLayout_6.addWidget(self.visualization_flag_button, 0, 0, 1, 1)

        self.visualization_flag_label = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.visualization_flag_label.sizePolicy().hasHeightForWidth())
        self.gridLayout_6.addWidget(self.visualization_flag_label, 0, 1, 1, 1)

        self.animation_flag_button = QtWidgets.QCheckBox(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.animation_flag_button.sizePolicy().hasHeightForWidth())
        self.gridLayout_6.addWidget(self.animation_flag_button, 1, 0, 1, 1)

        self.animation_flag_label = QtWidgets.QLabel(self.gridLayoutWidget_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.animation_flag_label.sizePolicy().hasHeightForWidth())
        self.gridLayout_6.addWidget(self.animation_flag_label, 1, 1, 1, 1)

        # Specimen Orientation Tab
        self.specimen_tab = QtWidgets.QWidget()
        self.specimen_tab.setObjectName("specimen_tab")
        self.InfoPages.addTab(self.specimen_tab, "")

        self.gridLayoutWidget_8 = QtWidgets.QWidget(self.specimen_tab)
        self.gridLayoutWidget_8.setGeometry(QtCore.QRect(20, 10, 300, 120))
        self.gridLayoutWidget_8.setObjectName("gridLayoutWidget_7")

        self.gridLayout_7 = QtWidgets.QGridLayout(self.gridLayoutWidget_8)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setObjectName("gridLayout_7")

        self.slope_offset_label = QtWidgets.QLabel(self.gridLayoutWidget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slope_offset_label.sizePolicy().hasHeightForWidth())
        self.gridLayout_7.addWidget(self.slope_offset_label, 0, 0, 1, 1)

        self.slope_offset_spinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slope_offset_spinBox.sizePolicy().hasHeightForWidth())
        self.gridLayout_7.addWidget(self.slope_offset_spinBox, 0, 1, 1, 1)

        # Log Tab
        self.log_tab = QtWidgets.QWidget()
        self.InfoPages.addTab(self.log_tab, "")
        self.log_tab.setObjectName("log_tab")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.LogFile = QtWidgets.QTextBrowser(self.log_tab)
        self.LogFile.setGeometry(QtCore.QRect(0, 0, 581, 336)) #change
        self.LogFile.setObjectName("LogFile")

        self.Visualization = QtWidgets.QGraphicsView(self.log_tab)
        self.Visualization.setGeometry(QtCore.QRect(580, 0, 321, 336)) #change
        self.Visualization.setObjectName("Visualization")

        ### Main Window
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 340, 820, 101)) #change
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.frame_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 12, 811, 91)) #change
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setHorizontalSpacing(10)
        self.gridLayout_2.setVerticalSpacing(10)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.frame_1 = QtWidgets.QFrame(self.centralwidget)
        self.frame_1.setGeometry(QtCore.QRect(10, 10, 911, 341)) #change
        self.frame_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_1.setObjectName("frame_1")
        self.formLayoutWidget = QtWidgets.QWidget(self.frame_1)
        self.formLayoutWidget.setGeometry(QtCore.QRect(0, 0, 901, 331)) #change
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.formLayoutWidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(5)
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setObjectName("gridLayout")

        # Dimensionality:
        self.dimensionality_label = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dimensionality_label.sizePolicy().hasHeightForWidth())
        self.dimensionality_label.setSizePolicy(sizePolicy)
        self.dimensionality_label.setObjectName("dimensionality_label")
        self.gridLayout.addWidget(self.dimensionality_label, 0, 0, 1, 1)

        self.two_d_button = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.two_d_button.setObjectName("two_d_button")
        self.gridLayout.addWidget(self.two_d_button, 0, 2, 1, 1)

        self.three_d_button = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.three_d_button.setChecked(True)
        self.three_d_button.setObjectName("three_d_button")
        self.gridLayout.addWidget(self.three_d_button, 0, 1, 1, 1)

        # Boxsize:
        self.BoxsizeLabel = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BoxsizeLabel.sizePolicy().hasHeightForWidth())
        self.BoxsizeLabel.setSizePolicy(sizePolicy)
        self.BoxsizeLabel.setObjectName("BoxsizeLabel")
        self.gridLayout.addWidget(self.BoxsizeLabel, 1, 0, 1, 1)

        self.box_sizeSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.box_sizeSpinBox.setProperty("value", 25.0)
        self.box_sizeSpinBox.setMaximum(1000)
        self.box_sizeSpinBox.setObjectName("box_sizeSpinBox")
        self.gridLayout.addWidget(self.box_sizeSpinBox, 1, 1, 1, 2)
        self.box_sizeSpinBox.valueChanged.connect(self.bandwidth_handler)

        self.box_size_y_Button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.box_size_y_Button.setChecked(False)
        self.box_size_y_Button.setObjectName("box_size_y_Button")
        self.gridLayout.addWidget(self.box_size_y_Button, 1, 3, 1, 2)
        self.box_size_y_Button.stateChanged.connect(self.button_handler)

        self.box_size_y_SpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.box_size_y_SpinBox.setProperty("value", 25.0)
        self.box_size_y_SpinBox.setMaximum(1000)
        self.box_size_y_SpinBox.setObjectName("box_size_y_SpinBox")
        self.gridLayout.addWidget(self.box_size_y_SpinBox, 1, 4, 1, 1)
        self.box_size_y_SpinBox.setEnabled(False)


        self.box_size_z_Button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.box_size_z_Button.setChecked(False)
        self.box_size_z_Button.setObjectName("box_size_z_Button")
        self.gridLayout.addWidget(self.box_size_z_Button, 1, 5, 1, 2)
        self.box_size_z_Button.stateChanged.connect(self.button_handler)

        self.box_size_z_SpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.box_size_z_SpinBox.setProperty("value", 25.0)
        self.box_size_z_SpinBox.setMaximum(1000)
        self.box_size_z_SpinBox.setObjectName("box_size_z_SpinBox")
        self.gridLayout.addWidget(self.box_size_z_SpinBox, 1, 6, 1, 1)
        self.box_size_z_SpinBox.setEnabled(False)




        # Resolution:
        self.resolutionLabel = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resolutionLabel.sizePolicy().hasHeightForWidth())
        self.resolutionLabel.setSizePolicy(sizePolicy)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.gridLayout.addWidget(self.resolutionLabel, 2, 0, 1, 1)

        self.resolutionSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.resolutionSpinBox.setDecimals(3)
        self.resolutionSpinBox.setMaximum(20.0)
        self.resolutionSpinBox.setMinimum(0.001)
        self.resolutionSpinBox.setValue(1)
        self.resolutionSpinBox.setSingleStep(0.01)
        self.resolutionSpinBox.setObjectName("resolutionSpinBox")
        self.gridLayout.addWidget(self.resolutionSpinBox, 2, 1, 1, 2)

        # Number of RVEs:
        self.NoRVEsLabel = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.NoRVEsLabel.sizePolicy().hasHeightForWidth())
        self.NoRVEsLabel.setSizePolicy(sizePolicy)
        self.NoRVEsLabel.setObjectName("NoRVEsLabel")
        self.gridLayout.addWidget(self.NoRVEsLabel, 3, 0, 1, 1)

        self.NoRVEsSpinBox = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.NoRVEsSpinBox.setMinimum(1)
        self.NoRVEsSpinBox.setMaximum(999)
        self.NoRVEsSpinBox.setSingleStep(1)
        self.NoRVEsSpinBox.setProperty("value", 1)
        self.NoRVEsSpinBox.setObjectName("NoRVEsSpinBox")
        self.gridLayout.addWidget(self.NoRVEsSpinBox, 3, 1, 1, 2)

        # Microstructure Phases:
        self.phases_label = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phases_label.sizePolicy().hasHeightForWidth())
        self.phases_label.setSizePolicy(sizePolicy)
        self.phases_label.setObjectName("phases_label")
        self.gridLayout.addWidget(self.phases_label, 4, 0, 1, 1)

        self.ferrite_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.ferrite_button.setChecked(False)
        self.ferrite_button.setObjectName("ferrite_button")
        self.gridLayout.addWidget(self.ferrite_button, 4, 1, 1, 2)
        self.ferrite_button.stateChanged.connect(self.phase_handler)

        self.martensite_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.martensite_button.setChecked(False)
        self.martensite_button.setObjectName("martensite_button")
        self.gridLayout.addWidget(self.martensite_button, 4, 4, 1, 1)
        self.martensite_button.stateChanged.connect(self.phase_handler)

        self.Pearlite_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.Pearlite_button.setChecked(False)
        self.Pearlite_button.setObjectName("Pearlite_button")
        self.gridLayout.addWidget(self.Pearlite_button, 4, 6, 1, 1)
        self.Pearlite_button.stateChanged.connect(self.phase_handler)

        self.Bainite_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.Bainite_button.setChecked(False)
        self.Bainite_button.setObjectName("Bainite_button")
        self.gridLayout.addWidget(self.Bainite_button, 4, 8, 1, 1)
        self.Bainite_button.stateChanged.connect(self.phase_handler)

        #Add
        self.Austenite_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.Austenite_button.setChecked(False)
        self.Austenite_button.setObjectName("Austenite_button")
        self.gridLayout.addWidget(self.Austenite_button, 4, 10, 1, 1)
        self.Austenite_button.stateChanged.connect(self.phase_handler)
        #Add

        # Phase Fraction:
        self.phase_fraction_label = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.phase_fraction_label.sizePolicy().hasHeightForWidth())
        self.phase_fraction_label.setSizePolicy(sizePolicy)
        self.phase_fraction_label.setObjectName("phase_fraction_label")
        self.gridLayout.addWidget(self.phase_fraction_label, 5, 0, 1, 1)

        self.ferriteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.ferriteSpinBox.setDecimals(2)
        self.ferriteSpinBox.setMaximum(1.0)
        self.ferriteSpinBox.setMinimum(0.01)
        self.ferriteSpinBox.setSingleStep(0.01)
        self.ferriteSpinBox.setEnabled(False)
        self.ferriteSpinBox.setObjectName("ferriteSpinBox")
        self.gridLayout.addWidget(self.ferriteSpinBox, 5, 1, 1, 2)

        self.martensiteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.martensiteSpinBox.setDecimals(2)
        self.martensiteSpinBox.setMaximum(1.0)
        self.martensiteSpinBox.setMinimum(0.01)
        self.martensiteSpinBox.setSingleStep(0.01)
        self.martensiteSpinBox.setEnabled(False)
        self.martensiteSpinBox.setObjectName("martensiteSpinBox")
        self.gridLayout.addWidget(self.martensiteSpinBox, 5, 4, 1, 1)

        self.pearliteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.pearliteSpinBox.setDecimals(2)
        self.pearliteSpinBox.setMaximum(1.0)
        self.pearliteSpinBox.setMinimum(0.01)
        self.pearliteSpinBox.setSingleStep(0.01)
        self.pearliteSpinBox.setEnabled(False)
        self.pearliteSpinBox.setObjectName("pearliteSpinBox")
        self.gridLayout.addWidget(self.pearliteSpinBox, 5, 6, 1, 1)

        self.bainiteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.bainiteSpinBox.setDecimals(2)
        self.bainiteSpinBox.setMaximum(1.0)
        self.bainiteSpinBox.setMinimum(0.01)
        self.bainiteSpinBox.setSingleStep(0.01)
        self.bainiteSpinBox.setEnabled(False)
        self.bainiteSpinBox.setObjectName("bainiteSpinBox")
        self.gridLayout.addWidget(self.bainiteSpinBox, 5, 8, 1, 1)

#Add
        self.austeniteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.austeniteSpinBox.setDecimals(2)
        self.austeniteSpinBox.setMaximum(1.0)
        self.austeniteSpinBox.setMinimum(0.01)
        self.austeniteSpinBox.setSingleStep(0.01)
        self.austeniteSpinBox.setEnabled(False)
        self.austeniteSpinBox.setObjectName("austeniteSpinBox")
        self.gridLayout.addWidget(self.austeniteSpinBox, 5, 10, 1, 2)
#Add
        
        # Inputfile path:
        self.filepath_label = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filepath_label.sizePolicy().hasHeightForWidth())
        self.filepath_label.setSizePolicy(sizePolicy)
        self.filepath_label.setObjectName("filepath_label")
        self.gridLayout.addWidget(self.filepath_label, 6, 0, 1, 1)

        self.lineEditFerrite = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEditFerrite.setEnabled(False)
        self.lineEditFerrite.setObjectName("lineEditFerrite")
        self.gridLayout.addWidget(self.lineEditFerrite, 6, 1, 1, 2)

        self.fileBrowserFerrite = QtWidgets.QPushButton(self.formLayoutWidget)
        self.fileBrowserFerrite.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserFerrite.sizePolicy().hasHeightForWidth())
        self.fileBrowserFerrite.setSizePolicy(sizePolicy)
        self.fileBrowserFerrite.setText("")
        self.fileBrowserFerrite.setIcon(icon)
        self.fileBrowserFerrite.setObjectName("fileBrowserFerrite")
        self.gridLayout.addWidget(self.fileBrowserFerrite, 6, 3, 1, 1)
        self.fileBrowserFerrite.clicked.connect(self.button_handler)

        self.lineEditMartensite = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEditMartensite.setEnabled(False)
        self.lineEditMartensite.setObjectName("lineEditMartensite")
        self.gridLayout.addWidget(self.lineEditMartensite, 6, 4, 1, 1)

        self.fileBrowserMartensite = QtWidgets.QPushButton(self.formLayoutWidget)
        self.fileBrowserMartensite.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserMartensite.sizePolicy().hasHeightForWidth())
        self.fileBrowserMartensite.setSizePolicy(sizePolicy)
        self.fileBrowserMartensite.setText("")
        self.fileBrowserMartensite.setIcon(icon)
        self.fileBrowserMartensite.setObjectName("fileBrowserMartensite")
        self.gridLayout.addWidget(self.fileBrowserMartensite, 6, 5, 1, 1)
        self.fileBrowserMartensite.clicked.connect(self.button_handler)

        self.lineEditPearlite = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEditPearlite.setEnabled(False)
        self.lineEditPearlite.setObjectName("lineEditPearlite")
        self.gridLayout.addWidget(self.lineEditPearlite, 6, 6, 1, 1)

        self.fileBrowser_Pearlite = QtWidgets.QPushButton(self.formLayoutWidget)
        self.fileBrowser_Pearlite.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowser_Pearlite.sizePolicy().hasHeightForWidth())
        self.fileBrowser_Pearlite.setSizePolicy(sizePolicy)
        self.fileBrowser_Pearlite.setText("")
        self.fileBrowser_Pearlite.setIcon(icon)
        self.fileBrowser_Pearlite.setObjectName("fileBrowser_Pearlite")
        self.gridLayout.addWidget(self.fileBrowser_Pearlite, 6, 7, 1, 1)
        self.fileBrowser_Pearlite.clicked.connect(self.button_handler)

        self.lineEditBainite = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEditBainite.setEnabled(False)
        self.lineEditBainite.setObjectName("lineEditBainite")
        self.gridLayout.addWidget(self.lineEditBainite, 6, 8, 1, 1)

        self.fileBrowserBainite = QtWidgets.QPushButton(self.formLayoutWidget)
        self.fileBrowserBainite.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserBainite.sizePolicy().hasHeightForWidth())
        self.fileBrowserBainite.setSizePolicy(sizePolicy)
        self.fileBrowserBainite.setText("")
        self.fileBrowserBainite.setIcon(icon)
        self.fileBrowserBainite.setObjectName("fileBrowserBainite")
        self.gridLayout.addWidget(self.fileBrowserBainite, 6, 9, 1, 1)
        self.fileBrowserBainite.clicked.connect(self.button_handler)
        
        #Add
        self.lineEditAustenite = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEditAustenite.setEnabled(False)
        self.lineEditAustenite.setObjectName("lineEditAustenite")
        self.gridLayout.addWidget(self.lineEditAustenite, 6, 10, 1, 1)

        self.fileBrowserAustenite = QtWidgets.QPushButton(self.formLayoutWidget)
        self.fileBrowserAustenite.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserAustenite.sizePolicy().hasHeightForWidth())
        self.fileBrowserAustenite.setSizePolicy(sizePolicy)
        self.fileBrowserAustenite.setText("")
        self.fileBrowserAustenite.setIcon(icon)
        self.fileBrowserAustenite.setObjectName("fileBrowserAustenite")
        self.gridLayout.addWidget(self.fileBrowserAustenite, 6, 11, 1, 1)
        self.fileBrowserAustenite.clicked.connect(self.button_handler)
        #Add

        # Microstructure Features:
        self.features_label = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.features_label.sizePolicy().hasHeightForWidth())
        self.features_label.setSizePolicy(sizePolicy)
        self.features_label.setObjectName("features_label")
        self.gridLayout.addWidget(self.features_label, 7, 0, 1, 1)

        self.Banding_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.Banding_button.setObjectName("Banding_button")
        self.gridLayout.addWidget(self.Banding_button, 7, 1, 1, 2)
        self.Banding_button.stateChanged.connect(self.features_handler)

        self.inclusions_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.inclusions_button.setObjectName("inclusions_button")
        self.gridLayout.addWidget(self.inclusions_button, 7, 4, 1, 1)
        self.inclusions_button.stateChanged.connect(self.features_handler)

        self.substructure_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.substructure_button.setObjectName("substructure_button")
        self.gridLayout.addWidget(self.substructure_button, 7, 6, 1, 1)
        self.substructure_button.stateChanged.connect(self.features_handler)

        self.roughness_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.roughness_button.setEnabled(False)
        self.roughness_button.setObjectName("roughness_button")
        self.gridLayout.addWidget(self.roughness_button, 7, 8, 1, 1)

        # Framework:
        self.framework_label = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.framework_label.sizePolicy().hasHeightForWidth())
        self.framework_label.setSizePolicy(sizePolicy)
        self.framework_label.setObjectName("framework_label")
        self.gridLayout_2.addWidget(self.framework_label, 0, 0, 1, 2)

        self.btngroup1 = QButtonGroup()
        self.btngroup2 = QButtonGroup()

        self.abaqus_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.btngroup1.addButton(self.abaqus_button)
        self.abaqus_button.setObjectName("abaqus_button")
        self.abaqus_button.setChecked(True)
        self.gridLayout_2.addWidget(self.abaqus_button, 0, 1)
        self.abaqus_button.toggled.connect(self.framework_handler)

        self.damask_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.btngroup1.addButton(self.damask_button)
        self.damask_button.setObjectName("damask_button")
        self.gridLayout_2.addWidget(self.damask_button, 0, 2)
        self.damask_button.toggled.connect(self.framework_handler)

        self.moose_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.btngroup1.addButton(self.moose_button)
        #self.moose_button.setChecked(True)
        self.moose_button.setObjectName("moose_button")
        self.gridLayout_2.addWidget(self.moose_button, 0, 3)
        self.moose_button.toggled.connect(self.framework_handler)

        self.boundary_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.boundary_label.setObjectName("boundary_label")
        self.gridLayout_2.addWidget(self.boundary_label, 0, 4, 1, 1)

        self.PBC_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.btngroup2.addButton(self.PBC_button)
        self.PBC_button.setObjectName("PBC_button")
        self.PBC_button.setChecked(True)
        self.gridLayout_2.addWidget(self.PBC_button, 0, 5, 1, 1)

        self.submodel_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.btngroup2.addButton(self.submodel_button)
        self.submodel_button.setObjectName("submodel_button")
        self.gridLayout_2.addWidget(self.submodel_button, 0, 6, 1, 1)

        self.xfem_button = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.xfem_button.setObjectName("xfem_button")
        self.xfem_button.setChecked(False)
        self.gridLayout_2.addWidget(self.xfem_button, 0, 7, 1, 1)

        # Element Type:
        self.element_type_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.element_type_label.setObjectName("element_type_label")
        self.gridLayout_2.addWidget(self.element_type_label, 1, 0, 1, 1)

        self.comboBox_element_type = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox_element_type.setObjectName("comboBox_element_type")
        self.comboBox_element_type.addItem("C3D4 (Abaqus Tet)")
        self.comboBox_element_type.addItem("C3D8 (Abaqus Hex)")
        self.gridLayout_2.addWidget(self.comboBox_element_type, 1, 1, 1, 3)

        # Smooth grainboundaries
        self.smoothing_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.smoothing_label.setObjectName("smoothing_label")
        self.gridLayout_2.addWidget(self.smoothing_label, 1, 4, 1, 1)

        self.smoothing_button = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.smoothing_button.setObjectName("smoothing_button")
        self.gridLayout_2.addWidget(self.smoothing_button, 1, 5, 1, 1)
        self.smoothing_button.setChecked(True)

        # Output directory:
        self.label_store_path = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_store_path.setObjectName("label_store_path")
        self.gridLayout_2.addWidget(self.label_store_path, 2, 0, 1, 1)

        self.lineEdit_store_path = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_store_path.setObjectName("lineEdit_store_path")
        self.gridLayout_2.addWidget(self.lineEdit_store_path, 2, 1, 1, 2)

        self.fileBrowserStore_path = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.fileBrowserStore_path.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileBrowserStore_path.sizePolicy().hasHeightForWidth())
        self.fileBrowserStore_path.setSizePolicy(sizePolicy)
        self.fileBrowserStore_path.setText("")
        self.fileBrowserStore_path.setIcon(icon)
        self.fileBrowserStore_path.setObjectName("fileBrowserStore_path")
        self.gridLayout_2.addWidget(self.fileBrowserStore_path, 2, 3, 1, 1)
        self.fileBrowserStore_path.clicked.connect(self.button_handler)

        # Progress Bar & Start Button
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 800, 281, 23))
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        self.StartButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartButton.setEnabled(True)
        self.StartButton.setGeometry(QtCore.QRect(300, 800, 141, 23))
        self.StartButton.setObjectName("StartButton")
        self.StartButton.clicked.connect(self.submit)

        ###
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionFiles = QtWidgets.QAction(MainWindow)
        self.actionFiles.setObjectName("actionFiles")

        self.retranslateUi(MainWindow)
        self.InfoPages.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)
        self.user_tab.setDisabled(True)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DRAGen - RVE Generator"))
        self.NoBand_label.setText(_translate("MainWindow", "Number of Bands"))
        self.band_lower_label.setText(_translate("MainWindow", "Band lower thickness boundary"))
        self.band_upper_label.setText(_translate("MainWindow", "Band upper thickness boundary"))
        self.band_filling_label.setText(_translate("MainWindow", "Band filling parameter"))
        self.band_orientation_label.setText(_translate("MainWindow", "Orientaion of band in RVE"))
        self.band_file_label.setText(_translate("MainWindow", "Input file for banded grains"))

        self.inclusion_file_label.setText(_translate("MainWindow", "Input file for inclusions/pores"))
        self.inclusion_ratio_label.setText(_translate("MainWindow", "phase ratio for inclusions/pores"))

        self.substructure_filemode_radio.setText(_translate("MainWindow", "File Mode"))
        self.substructure_user_mode_radio.setText(_translate("MainWindow", "User Mode"))
        self.label_block_thicknes_file.setText(_translate("MainWindow", "block thickness (sigma)"))
        self.label_save_result_file.setText(_translate("MainWindow", "save result as:"))
        self.label_packet_size_file.setText(_translate("MainWindow", "Packet size (sigma)"))
        self.label_max_block_thickness_file.setText(_translate("MainWindow", "maximum blockthickness"))
        self.label_decreasing_fact_file.setText(_translate("MainWindow", "decreasing factor"))
        self.label_min_block_thickness_file.setText(_translate("MainWindow", "minimum blockthickness"))
        self.label_substructure_file.setText(_translate("MainWindow", "Substructure File"))

        self.label_max_block_thickness_user.setText(_translate("MainWindow", "maximum blockthickness"))
        self.label_decreasing_fact_user.setText(_translate("MainWindow", "decreasing factor"))
        self.label_packet_eq_d_user.setText(_translate("MainWindow", "Packet equiv_d"))
        self.label_min_block_thickness_user.setText(_translate("MainWindow", "minimum blockthickness"))
        self.label_save_result_user.setText(_translate("MainWindow", "save result as:"))
        self.label_block_thickness_user.setText(_translate("MainWindow", "block thickness"))
        self.label_block_thickness_sigma_user.setText(_translate("MainWindow", "block thickness (sigma)"))
        self.label_packet_size_user.setText(_translate("MainWindow", "Packet size (sigma)"))
        self.label_circularity_user.setText(_translate("MainWindow", "circulatrity (packets)"))

        self.visualization_flag_label.setText(_translate("MainWindow", "Plot figures of RSA and Tesselation"))
        self.animation_flag_label.setText(_translate("MainWindow", "Plot figures of Banding RSA"))

        self.slope_offset_label.setText(_translate("MainWindow", "enter the angle for your specimenorientation\n"
                                                                 "0 means rolling direction along x-axis\n"
                                                                 "90 means rolling direction along y-axis"))

        self.InfoPages.setTabText(self.InfoPages.indexOf(self.status_tab), _translate("MainWindow", "Status"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.banding_tab), _translate("MainWindow", "Banding Feature"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.inclusion_tab), _translate("MainWindow", "Inclusion Feature"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.substructure), _translate("MainWindow", "Substructure Feature"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.file_tab), _translate("MainWindow", "File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.user_tab), _translate("MainWindow", "User"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.visualization_tab), _translate("MainWindow", "Visualization"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.specimen_tab), _translate("MainWindow", "Specimen Orient."))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.log_tab), _translate("MainWindow", "Log"))

        self.StartButton.setText(_translate("MainWindow", "Start Generation"))
        self.box_size_y_Button.setText(_translate("MainWindow", "y: "))
        self.box_size_z_Button.setText(_translate("MainWindow", "z: "))
        self.element_type_label.setText(_translate("MainWindow", "Element Type"))
        self.smoothing_label.setText(_translate("MainWindow", "Smooth Grainboundaries"))
        self.damask_button.setText(_translate("MainWindow", "Damask"))
        self.abaqus_button.setText(_translate("MainWindow", "Abaqus"))
        self.moose_button.setText(_translate("MainWindow", "Moose"))
        self.framework_label.setText(_translate("MainWindow", "Framework:"))
        self.boundary_label.setText(_translate("MainWindow", "BC:"))
        self.PBC_button.setText(_translate("MainWindow", "periodic"))
        self.submodel_button.setText(_translate("MainWindow", "submodel"))
        self.xfem_button.setText(_translate("MainWindow", "xfem"))
        self.label_store_path.setText(_translate("MainWindow", "Output directory"))
        self.lineEdit_store_path.setText(_translate("MainWindow", "C:\\temp"))
        self.Banding_button.setText(_translate("MainWindow", "Banding"))
        self.ferrite_button.setText(_translate("MainWindow", "Ferrite"))
        self.martensite_button.setText(_translate("MainWindow", "Martensite"))
        self.features_label.setText(_translate("MainWindow", "Microstructure Features:"))
        self.NoRVEsLabel.setText(_translate("MainWindow", "Number of RVEs:"))
        self.resolutionLabel.setText(_translate("MainWindow", "Resolution:"))
        self.BoxsizeLabel.setText(_translate("MainWindow", "Boxsize: "))

        self.phases_label.setText(_translate("MainWindow", "Microstructure Phases:"))
        self.dimensionality_label.setText(_translate("MainWindow", "Dimensionality:"))
        self.filepath_label.setText(_translate("MainWindow", "Inputfile path:"))
        self.roughness_button.setText(_translate("MainWindow", "Surface Roughness"))
        self.two_d_button.setText(_translate("MainWindow", "2D"))
        self.three_d_button.setText(_translate("MainWindow", "3D"))
        self.substructure_button.setText(_translate("MainWindow", "Substructure"))
        self.inclusions_button.setText(_translate("MainWindow", "Inclusions/Pores"))
        self.Pearlite_button.setText(_translate("MainWindow", "Pearlite"))
        self.Bainite_button.setText(_translate("MainWindow", "Bainite"))
        self.Austenite_button.setText(_translate("MainWindow", "Austenite"))  #Add
        self.phase_fraction_label.setText(_translate("MainWindow", "Phase Fraction:"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionFiles.setText(_translate("MainWindow", "Files"))

    def progress_event(self, progress):
        self.progressBar.setValue(progress)

    def phase_handler(self, state):
        if self.MainWindow.sender() == self.ferrite_button:
            if state == Qt.Checked:
                self.ferriteSpinBox.setEnabled(True)
                self.lineEditFerrite.setEnabled(True)
                self.fileBrowserFerrite.setEnabled(True)
            else:
                self.ferriteSpinBox.setEnabled(False)
                self.lineEditFerrite.setEnabled(False)
                self.fileBrowserFerrite.setEnabled(False)

        if self.MainWindow.sender() == self.martensite_button:
            if state == Qt.Checked:
                self.martensiteSpinBox.setEnabled(True)
                self.lineEditMartensite.setEnabled(True)
                self.fileBrowserMartensite.setEnabled(True)
            else:
                self.martensiteSpinBox.setEnabled(False)
                self.lineEditMartensite.setEnabled(False)
                self.fileBrowserMartensite.setEnabled(False)
        
        if self.MainWindow.sender() == self.Pearlite_button:
            if state == Qt.Checked:
                self.pearliteSpinBox.setEnabled(True)
                self.lineEditPearlite.setEnabled(True)
                self.fileBrowser_Pearlite.setEnabled(True)
            else:
                self.pearliteSpinBox.setEnabled(False)
                self.lineEditPearlite.setEnabled(False)
                self.fileBrowser_Pearlite.setEnabled(False)
        
        if self.MainWindow.sender() == self.Bainite_button:
            if state == Qt.Checked:
                self.bainiteSpinBox.setEnabled(True)
                self.lineEditBainite.setEnabled(True)
                self.fileBrowserBainite.setEnabled(True)
            else:
                self.bainiteSpinBox.setEnabled(False)
                self.lineEditBainite.setEnabled(False)
                self.fileBrowserBainite.setEnabled(False)
#Add
        if self.MainWindow.sender() == self.Austenite_button:
            if state == Qt.Checked:
                self.austeniteSpinBox.setEnabled(True)
                self.lineEditAustenite.setEnabled(True)
                self.fileBrowserAustenite.setEnabled(True)
            else:
                self.austeniteSpinBox.setEnabled(False)
                self.lineEditAustenite.setEnabled(False)
                self.fileBrowserAustenite.setEnabled(False)
#Add
    def bandwidth_handler(self):
        min_thickness = 1/self.resolutionSpinBox.value()
        if self.MainWindow.sender() == self.band_lowerSpinBox:
            self.band_upperSpinBox.setMinimum(self.band_lowerSpinBox.value()+min_thickness)
        elif self.MainWindow.sender() == self.band_upperSpinBox:
            self.band_lowerSpinBox.setMaximum(self.band_upperSpinBox.value()-min_thickness)

    def button_handler(self):
        if self.MainWindow.sender() == self.fileBrowserFerrite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditFerrite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserMartensite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditMartensite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowser_Pearlite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditPearlite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserBainite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditBainite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserAustenite: #Add
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditAustenite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserBand:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditBand.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserInclusion:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.pkl")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditInclusion.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserStore_path:
            file_path = QFileDialog.getExistingDirectory()
            self.lineEdit_store_path.setText(file_path)
        elif self.MainWindow.sender() == self.substructure_file_browser_file:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.substructure_file_lineEdit_file.setText(file_path[0])

        elif self.MainWindow.sender() == self.box_size_y_Button and self.three_d_button.isChecked():
            if self.box_size_y_Button.isChecked():
                self.box_size_y_SpinBox.setEnabled(True)
            else:
                self.box_size_y_SpinBox.setDisabled(True)

        elif self.MainWindow.sender() == self.box_size_z_Button and self.three_d_button.isChecked():
            if self.box_size_z_Button.isChecked():
                self.box_size_z_SpinBox.setEnabled(True)
            else:
                self.box_size_z_SpinBox.setDisabled(True)

    def features_handler(self, state):
        if self.MainWindow.sender() == self.Banding_button:
            if state == Qt.Checked:
                self.InfoPages.setCurrentIndex(1)
                self.banding_tab.setEnabled(True)
            else:
                self.banding_tab.setDisabled(True)
        elif self.MainWindow.sender() == self.inclusions_button:
            if state == Qt.Checked:
                self.InfoPages.setCurrentIndex(2)
                self.inclusion_tab.setEnabled(True)
            else:
                self.inclusion_tab.setDisabled(True)
        elif self.MainWindow.sender() == self.substructure_button:
            if state == Qt.Checked:
                self.InfoPages.setCurrentIndex(3)
                self.substructure.setEnabled(True)
            else:
                self.substructure.setDisabled(True)


    def widget_handler(self):
        if self.substructure_filemode_radio.isChecked():
            self.file_tab.setEnabled(True)
            self.tabWidget.setCurrentIndex(0)
            self.user_tab.setDisabled(True)
        elif self.substructure_user_mode_radio.isChecked():
            self.user_tab.setEnabled(True)
            self.tabWidget.setCurrentIndex(1)
            self.file_tab.setDisabled(True)

    def substructure_file_checkbox_handler(self):
        if self.MainWindow.sender() == self.substructure_packet_size_checkBox_file:
            self.substructure_packet_size_SpinBox_file.setEnabled(self.substructure_packet_size_checkBox_file.isChecked())
        elif self.MainWindow.sender() == self.substructure_block_thickness_checkBox_file:
            self.substructure_block_thickness_SpinBox_file.setEnabled(self.substructure_block_thickness_checkBox_file.isChecked())
        elif self.MainWindow.sender() == self.substructure_decreasing_fact_checkBox_file:
            self.substructure_decreasing_fact_SpinBox_file.setEnabled(self.substructure_decreasing_fact_checkBox_file.isChecked())
        elif self.MainWindow.sender() == self.substructure_min_block_thickness_checkBox_file:
            self.substructure_min_block_thickness_SpinBox_file.setEnabled(self.substructure_min_block_thickness_checkBox_file.isChecked())
        elif self.MainWindow.sender() == self.substructure_max_block_thickness_checkBox_file:
            self.substructure_max_block_thickness_SpinBox_file.setEnabled(self.substructure_max_block_thickness_checkBox_file.isChecked())
        elif self.MainWindow.sender() == self.substructure_save_result_checkBox_file:
            self.substructure_save_result_lineEdit_file.setEnabled(self.substructure_save_result_checkBox_file.isChecked())

    def substructure_user_checkbox_handler(self):
        if self.MainWindow.sender() == self.substructure_packet_size_checkBox_user:
            self.substructure_packet_size_SpinBox_user.setEnabled(self.substructure_packet_size_checkBox_user.isChecked())
        elif self.MainWindow.sender() == self.substructure_circularity_checkBox_user:
            self.substructure_circularity_SpinBox_user.setEnabled(self.substructure_circularity_checkBox_user.isChecked())
        elif self.MainWindow.sender() == self.substructure_decreasing_fact_checkBox_user:
            self.substructure_decreasing_fact_SpinBox_user.setEnabled(self.substructure_decreasing_fact_checkBox_user.isChecked())
        elif self.MainWindow.sender() == self.substructure_block_thickness_sigma_checkBox_user:
            self.substructure_block_thickness_sigma_SpinBox_user.setEnabled(self.substructure_block_thickness_sigma_checkBox_user.isChecked())
        elif self.MainWindow.sender() == self.substructure_min_block_thickness_checkBox_user:
            self.substructure_min_block_thickness_SpinBox_user.setEnabled(self.substructure_min_block_thickness_checkBox_user.isChecked())
        elif self.MainWindow.sender() == self.substructure_max_block_thickness_checkBox_user:
            self.substructure_max_block_thickness_SpinBox_user.setEnabled(self.substructure_max_block_thickness_checkBox_user.isChecked())
        elif self.MainWindow.sender() == self.substructure_save_result_checkBox_user:
            self.substructure_save_result_lineEdit_user.setEnabled(self.substructure_save_result_checkBox_user.isChecked())

    def framework_handler(self):
        if self.moose_button.isChecked():
            self.comboBox_element_type.setEnabled(True)
            self.boundary_label.hide()
            self.PBC_button.hide()
            self.submodel_button.hide()
            self.comboBox_element_type.clear()
            self.comboBox_element_type.addItem("HEX8 (Moose)")
            self.smoothing_button.show()
            self.smoothing_label.show()
        elif self.damask_button.isChecked():
            self.comboBox_element_type.setEnabled(False)
            self.boundary_label.hide()
            self.PBC_button.hide()
            self.submodel_button.hide()
            self.comboBox_element_type.clear()
            self.smoothing_button.hide()
            self.smoothing_label.hide()
        elif self.abaqus_button.isChecked():
            self.comboBox_element_type.setEnabled(True)
            self.boundary_label.show()
            self.PBC_button.show()
            self.submodel_button.show()
            self.comboBox_element_type.clear()
            self.comboBox_element_type.addItem("C3D4 (Abaqus Tet)")
            self.comboBox_element_type.addItem("C3D8 (Abaqus Hex)")
            self.smoothing_button.show()
            self.smoothing_label.show()


    def submit(self):

        ARGS = {'root': None, 'box_size': None, 'box_size_y': None, 'box_size_z': None, 'resolution': None,
                'number_of_rves': 0, 'dimension': 3, 'phases': list(), 'abaqus_flag': False, 'damask_flag': False,
                'moose_flag': False, 'anim_flag': None, 'phase2iso_flag': True, 'xfem_flag': False, 'pbc_flag': False,
                'submodel_flag': False, 'element_type': None, 'slope_offset': 0, 'smoothing': True,
                'number_of_bands': 0, 'lower_band_bound': None, 'upper_band_bound': None, 'band_orientation': None,
                'band_filling': None, 'visualization_flag': None,
                'file_dict': {}, 'phase_ratio': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7:0}, #change
                'subs_flag': False, 'subs_file_flag': False,
                'subs_file': None, 'orientation_relationship': None, 'subrun': None, 'pag_file': None, 'equiv_d': None,
                'circularity': 1, 'p_sigma': 0.1, 'block_file': None, 't_mu': None, 'b_sigma': 0.1,
                'decreasing_factor': 0.95, 'lower': None, 'upper': None, 'plt_name': None, 'save': True, 'plot': None,
                'filename': 'substruct_data.csv', 'gui_flag': True,
                'files': {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}}

        if self.two_d_button.isChecked():
            ARGS['dimension'] = 2
            dimension_flag = True
        else:
            ARGS['dimension'] = 3
            dimension_flag = True

        ARGS['box_size'] = self.box_sizeSpinBox.value()
        if self.box_size_y_Button.isChecked():
            ARGS['box_size_y'] = self.box_size_y_SpinBox.value()
        if self.box_size_z_Button.isChecked():
            ARGS['box_size_z'] = self.box_size_z_SpinBox.value()

        ARGS['resolution'] = self.resolutionSpinBox.value()
        ARGS['number_of_rves'] = self.NoRVEsSpinBox.value()
        ARGS['slope_offset'] = self.slope_offset_spinBox.value()

        if not self.ferrite_button.isChecked() \
                and not self.martensite_button.isChecked() \
                and not self.Pearlite_button.isChecked() \
                and not self.Bainite_button.isChecked() \
                and not self.Austenite_button.isChecked(): #Add, and previous line changed
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please choose at least one phase!")
            msg.setInformativeText("Check one of the checkboxes stating\nFerrite, Martensite, Pearlite, Bainite or Austenite") #Change
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        if self.ferrite_button.isChecked():
            file1 = self.lineEditFerrite.text()
            phase1_ratio = self.ferriteSpinBox.value()

            if len(file1) > 0:
                ARGS['phases'].append('Ferrite')
                ARGS['files'][1] = file1
                ARGS['phase_ratio'][1] = phase1_ratio
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for Ferrite!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return

        if self.martensite_button.isChecked():
            file2 = self.lineEditMartensite.text()
            phase2_ratio = self.martensiteSpinBox.value()

            if len(file2) > 0:
                ARGS['phases'].append('Martensite')
                ARGS['files'][2] = file2
                ARGS['phase_ratio'][2] = phase2_ratio
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for Martensite!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return

        if self.Pearlite_button.isChecked():
            file3 = self.lineEditPearlite.text()
            phase3_ratio = self.pearliteSpinBox.value()

            if len(file3) > 0:
                ARGS['phases'].append('Pearlite')
                ARGS['files'][3] = file3
                ARGS['phase_ratio'][3] = phase3_ratio
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for Pearlite!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return

        if self.Bainite_button.isChecked():
            file4 = self.lineEditBainite.text()
            phase4_ratio = self.bainiteSpinBox.value()

            if len(file4) > 0:
                ARGS['phases'].append('Bainite')
                ARGS['files'][4] = file4
                ARGS['phase_ratio'][4] = phase4_ratio
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for Bainite!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
#Add->
        if self.Austenite_button.isChecked():
            file5 = self.lineEditAustenite.text()
            phase5_ratio = self.austeniteSpinBox.value()

            if len(file5) > 0:
                ARGS['phases'].append('Austenite')
                ARGS['files'][5] = file5
                ARGS['phase_ratio'][5] = phase5_ratio
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for Austenite!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
#Add<-
#Change->                  
        if self.inclusions_button.isChecked():
            file6 = self.lineEditInclusion.text()
            phase6_ratio = self.inclusionSpinBox.value()
            if len(file6) > 0:
                ARGS['phases'].append('Inclusions')
                ARGS['phase_ratio'][6] = phase6_ratio
                ARGS['files'][6] = file6
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for the inclusions/pores!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            # TODO: hier fehlt noch die void/solid option

        if self.Banding_button.isChecked():
            file7 = self.lineEditBand.text()
            if len(file7) > 0:
                ARGS['number_of_bands'] = self.NoBandsSpinBox.value()
                ARGS['lower_band_bound'] = self.band_lowerSpinBox.value()
                ARGS['upper_band_bound'] = self.band_upperSpinBox.value()
                ARGS['phases'].append('Bands')
                ARGS['phase_ratio'][7] = 0
                ARGS['files'][7] = file7
                ARGS['band_filling'] = self.band_fillingSpinBox.value()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please select a data input file for the inclusions/pores!")
                msg.setWindowTitle("Error")
                msg.exec_()
                return

            if self.BandOrientation_XY.isChecked():
                ARGS['band_orientation'] = 'xy'
            elif self.BandOrientation_XZ.isChecked():
                ARGS['band_orientation'] = 'xz'
            elif self.BandOrientation_YZ.isChecked():
                ARGS['band_orientation'] = 'yz'
#change<-

        sum_ratio = sum(ARGS['phase_ratio'].values())

        if sum_ratio != 1 and sum_ratio != 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("The sum of all phase ratios must equal to 1!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        elif sum_ratio == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please enter phase ratios for selected phases!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        if self.substructure_button.isChecked():
            if self.substructure_filemode_radio.isChecked():
                ARGS['subs_file_flag'] = True
                ARGS['subs_flag'] = True
                ARGS['subs_file'] = self.substructure_file_lineEdit_file.text()
                ARGS['equiv_d'] = None
                ARGS['t_mu'] = None
                ARGS['circularity'] = 1.0
                ARGS['decreasing_factor'] = 0.95
                if self.substructure_decreasing_fact_checkBox_file.isChecked():
                    ARGS['decreasing_factor'] = self.substructure_decreasing_fact_SpinBox_file.value()
                ARGS['p_sigma'] = 0.01
                if self.substructure_packet_size_checkBox_file.isChecked():
                    ARGS['p_sigma'] = self.substructure_packet_size_SpinBox_file.value()
                ARGS['b_sigma'] = 0.01
                if self.substructure_block_thickness_checkBox_file.isChecked():
                    ARGS['b_sigma'] = self.substructure_block_thickness_SpinBox_file.value()
                ARGS['lower'] = None
                if self.substructure_min_block_thickness_checkBox_file.isChecked():
                    ARGS['lower'] = self.substructure_min_block_thickness_SpinBox_file.value()
                ARGS['upper'] = None
                if self.substructure_max_block_thickness_checkBox_file.isChecked():
                    ARGS['upper'] = self.substructure_max_block_thickness_SpinBox_file.value()
                ARGS['save'] = True
                if self.substructure_save_result_checkBox_file.isChecked():
                    ARGS['save'] = True
                    ARGS['filename'] = self.substructure_save_result_lineEdit_file.text()
                else:
                    ARGS['save'] = False
            else:
                ARGS['subs_flag'] = True
                ARGS['equiv_d'] = self.substructure_packet_eq_d_SpinBox_user.value()
                ARGS['t_mu'] = self.substructure_block_thickness_SpinBox_user.value()
                ARGS['subs_file_flag'] = False
                ARGS['decreasing_factor'] = 0.95
                if self.substructure_decreasing_fact_checkBox_user.isChecked():
                    ARGS['decreasing_factor'] = self.substructure_decreasing_fact_SpinBox_user.value()
                ARGS['circularity'] = 1.0
                if self.substructure_circularity_checkBox_user.isChecked():
                    ARGS['circularity'] = self.substructure_circularity_SpinBox_user.value()
                ARGS['p_sigma'] = 0.01
                if self.substructure_packet_size_checkBox_user.isChecked():
                    ARGS['p_sigma'] = self.substructure_packet_size_SpinBox_user.value()
                ARGS['b_sigma'] = 0.01
                if self.substructure_block_thickness_sigma_checkBox_user.isChecked():
                    ARGS['b_sigma'] = self.substructure_block_thickness_sigma_SpinBox_user.value()
                ARGS['lower'] = None
                if self.substructure_min_block_thickness_checkBox_user.isChecked():
                    ARGS['lower'] = self.substructure_min_block_thickness_SpinBox_user.value()
                ARGS['upper'] = None
                if self.substructure_max_block_thickness_checkBox_user.isChecked():
                    ARGS['upper'] = self.substructure_max_block_thickness_SpinBox_user.value()
                ARGS['save'] = True
                if self.substructure_save_result_checkBox_user.isChecked():
                    ARGS['save'] = True
                    ARGS['filename'] = self.substructure_save_result_lineEdit_user.text()
                else:
                    ARGS['save'] = False
                ARGS['subs_file'] = None

        # if self.roughness_button.isChecked(): # TODO: Nach release hinzufügen?

        if self.substructure_button.isChecked():
            ARGS['subs_flag'] = True
            if self.substructure_filemode_radio.isChecked():
                ARGS['subs_file_flag'] = True
                ARGS['subs_file'] = self.substructure_file_lineEdit_file.text()
                ARGS['p_sigma'] = self.substructure_packet_size_SpinBox_file.value()
                ARGS['b_sigma'] = self.substructure_block_thickness_SpinBox_file.value()
                ARGS['decreasing_factor'] = self.substructure_decreasing_fact_SpinBox_file.value()
                ARGS['lower'] = self.substructure_min_block_thickness_SpinBox_file.value()
                ARGS['upper'] = self.substructure_max_block_thickness_SpinBox_file.value()
                ARGS['save'] = self.substructure_save_result_lineEdit_file.text()
            elif self.substructure_user_mode_radio.isChecked():
                ARGS['subs_file_flag'] = False # TODO: @Linghao please check this
                ARGS['equiv_d'] = self.substructure_packet_eq_d_SpinBox_user.value()
                ARGS['p_sigma'] = self.substructure_packet_size_SpinBox_user.value()
                ARGS['circularity'] = self.substructure_circularity_SpinBox_user.value()
                ARGS['block_thickness'] = None  # TODO: @Linghao please check this

                ARGS['decreasing_factor'] = self.substructure_decreasing_fact_SpinBox_user.value()
                ARGS['b_sigma'] = self.substructure_block_thickness_SpinBox_user.value()
                ARGS['lower'] = self.substructure_min_block_thickness_SpinBox_user.value()
                ARGS['upper'] = self.substructure_max_block_thickness_SpinBox_user.value()
                ARGS['save'] = self.substructure_save_result_lineEdit_user.text()

        if self.abaqus_button.isChecked():
            ARGS['abaqus_flag'] = True
            element_type_dict = {0: 'C3D4', 1: 'C3D8'}
            ARGS['element_type'] = element_type_dict.get(self.comboBox_element_type.currentIndex())
            ARGS['submodel_flag'] = self.submodel_button.isChecked()
            ARGS['pbc_flag'] = self.PBC_button.isChecked()
            ARGS['xfem_flag'] = self.xfem_button.isChecked()
            ARGS['smoothing'] = self.smoothing_button.isChecked()
        elif self.damask_button.isChecked():
            ARGS['damask_flag'] = True
        else:
            element_type_dict = {0: 'HEX8'}
            ARGS['moose_flag'] = True
            ARGS['element_type'] = element_type_dict.get(self.comboBox_element_type.currentIndex())
            ARGS['smoothing'] = self.smoothing_button.isChecked()

        ARGS['root'] = self.lineEdit_store_path.text()
        if len(ARGS['root']) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please choose a directory to store the Output data in!")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        else:
            store_path_flag = True

        if self.visualization_flag_button.isChecked():
            ARGS['visualization_flag'] = True
        else:
            ARGS['visualization_flag'] = False

        if self.animation_flag_button.isChecked():
            ARGS['animation_flag'] = True
        else:
            ARGS['animation_flag'] = False

        if len(ARGS['files'].values()) > 0:
            import_flag = True
        else:
            import_flag = False

        if dimension_flag and store_path_flag and import_flag:
            self.textBrowser.setText('Staring the generation of {} RVEs'.format(ARGS['number_of_rves']))
            self.thread = QThread()
            self.worker = Worker(ARGS)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.progress_event)
            self.worker.info_box.connect(self.textBrowser.setText)
            self.thread.start()
            self.StartButton.setEnabled(False)
            self.textBrowser.setText(' Generation running...')
            self.thread.finished.connect(lambda: self.StartButton.setEnabled(True))
            self.thread.finished.connect(lambda: self.textBrowser.setText(' The generation has finished!'))


if __name__ == "__main__":
    "test"
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QIcon


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("Main Window")
        MainWindow.setWindowTitle("DRAGen - RVE Generator")
        MainWindow.resize(735, 826)
        self.thumbnail_path = sys.argv[0][:-10] + "\\dragen\\thumbnails\\"
        MainWindow.setWindowIcon(QIcon(self.thumbnail_path + '\\Drache.ico'))

        # Definition of Containers:
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 460, 711, 321))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 709, 319))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.InfoPages = QtWidgets.QTabWidget(self.scrollAreaWidgetContents)
        self.InfoPages.setEnabled(True)
        self.InfoPages.setGeometry(QtCore.QRect(-10, 10, 721, 311))
        self.InfoPages.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.InfoPages.setUsesScrollButtons(True)
        self.InfoPages.setTabsClosable(False)
        self.InfoPages.setMovable(False)
        self.InfoPages.setTabBarAutoHide(False)
        self.InfoPages.setObjectName("InfoPages")

        ### Tab window
        # Banding Feature Tab
        self.banding_tab = QtWidgets.QWidget()
        self.banding_tab.setObjectName("banding_tab")
        self.InfoPages.addTab(self.banding_tab, "")
        self.banding_tab.setEnabled(False)

        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.banding_tab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(20, 10, 161, 80))
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
        self.gridLayout_3.addWidget(self.NoBand_label, 0, 0, 1, 1)

        self.NoBandsSpinBox = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.NoBandsSpinBox.sizePolicy().hasHeightForWidth())
        self.NoBandsSpinBox.setSizePolicy(sizePolicy)
        self.NoBandsSpinBox.setObjectName("NoBandsSpinBox")
        self.gridLayout_3.addWidget(self.NoBandsSpinBox, 0, 1, 1, 1)
        
        self.band_thickness_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_thickness_label.sizePolicy().hasHeightForWidth())
        self.band_thickness_label.setSizePolicy(sizePolicy)
        self.band_thickness_label.setObjectName("band_thickness_label")
        self.gridLayout_3.addWidget(self.band_thickness_label, 1, 0, 1, 1)

        self.band_thicknessSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.band_thicknessSpinBox.sizePolicy().hasHeightForWidth())
        self.band_thicknessSpinBox.setSizePolicy(sizePolicy)
        self.band_thicknessSpinBox.setObjectName("band_thicknessSpinBox")
        self.gridLayout_3.addWidget(self.band_thicknessSpinBox, 1, 1, 1, 1)
        
        # Substructure Feature Tab
        self.substructure = QtWidgets.QWidget()
        self.substructure.setObjectName("substructure")
        self.InfoPages.addTab(self.substructure, "")
        self.substructure.setEnabled(False)

        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.substructure)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 0, 160, 21))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")

        self.grid_mode = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.grid_mode.setContentsMargins(0, 0, 0, 0)
        self.grid_mode.setObjectName("grid_mode")

        self.substructure_filemode_radio = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.substructure_filemode_radio.setChecked(True)
        self.substructure_filemode_radio.setObjectName("substructure_filemode_radio")
        self.grid_mode.addWidget(self.substructure_filemode_radio, 0, 0, 1, 1)
        self.substructure_filemode_radio.toggled.connect(self.widget_handler)

        self.substructure_user_mode_radio = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.substructure_user_mode_radio.setChecked(False)
        self.substructure_user_mode_radio.setObjectName("substructure_user_mode_radio")
        self.grid_mode.addWidget(self.substructure_user_mode_radio, 0, 1, 1, 1)
        self.substructure_user_mode_radio.toggled.connect(self.widget_handler)

        self.tabWidget = QtWidgets.QTabWidget(self.substructure)
        self.tabWidget.setGeometry(QtCore.QRect(6, 29, 721, 261))
        self.tabWidget.setObjectName("tabWidget")

        # Substructure Feature Tab -> File Tab
        self.file_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(self.file_tab, "")
        self.file_tab.setObjectName("file_tab")

        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.file_tab)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(0, 0, 421, 231))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")

        self.grid_file = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.grid_file.setContentsMargins(0, 0, 0, 0)
        self.grid_file.setObjectName("grid_file")

        self.label_substructure_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_substructure_file.setObjectName("label_substructure_file")
        self.grid_file.addWidget(self.label_substructure_file, 0, 0, 1, 1)

        self.substructure_file_lineEdit_file = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.substructure_file_lineEdit_file.setObjectName("substructure_file_lineEdit_file")
        self.grid_file.addWidget(self.substructure_file_lineEdit_file, 0, 1, 1, 1)

        self.substructure_file_browser_file = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.substructure_file_browser_file.sizePolicy().hasHeightForWidth())
        self.substructure_file_browser_file.setSizePolicy(sizePolicy)
        self.substructure_file_browser_file.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./dragen/thumbnails/Folder-Generic-Silver-icon.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        self.substructure_file_browser_file.setIcon(icon)
        self.substructure_file_browser_file.setObjectName("substructure_file_browser_file")
        self.grid_file.addWidget(self.substructure_file_browser_file, 0, 2, 1, 1)

        self.label_pack_size_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_pack_size_file.setObjectName("label_pack_size_file")
        self.grid_file.addWidget(self.label_pack_size_file, 1, 0, 1, 1)

        self.substructure_pack_size_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_4)
        self.substructure_pack_size_SpinBox_file.setEnabled(False)
        self.substructure_pack_size_SpinBox_file.setObjectName("substructure_pack_size_SpinBox")
        self.grid_file.addWidget(self.substructure_pack_size_SpinBox_file, 1, 1, 1, 1)

        self.substructure_packet_size_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_4)
        self.substructure_packet_size_checkBox_file.setText("")
        self.substructure_packet_size_checkBox_file.setObjectName("substructure_packet_size_checkBox_file")
        self.grid_file.addWidget(self.substructure_packet_size_checkBox_file, 1, 2, 1, 1)

        self.label_block_thicknes_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_block_thicknes_file.setObjectName("label_block_thicknes_file")
        self.grid_file.addWidget(self.label_block_thicknes_file, 2, 0, 1, 1)

        self.substructure_block_thickness_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_4)
        self.substructure_block_thickness_SpinBox_file.setEnabled(False)
        self.substructure_block_thickness_SpinBox_file.setObjectName("substructure_block_thickness_SpinBox_file")
        self.grid_file.addWidget(self.substructure_block_thickness_SpinBox_file, 2, 1, 1, 1)

        self.substructure_block_thickness_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_4)
        self.substructure_block_thickness_checkBox_file.setText("")
        self.substructure_block_thickness_checkBox_file.setObjectName("substructure_block_thickness_checkBox_file")
        self.grid_file.addWidget(self.substructure_block_thickness_checkBox_file, 2, 2, 1, 1)

        self.label_decreasing_fact_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_decreasing_fact_file.setObjectName("label_decreasing_fact_file")
        self.grid_file.addWidget(self.label_decreasing_fact_file, 3, 0, 1, 1)

        self.substructure_decreasing_fact_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_4)
        self.substructure_decreasing_fact_SpinBox_file.setEnabled(False)
        self.substructure_decreasing_fact_SpinBox_file.setObjectName("substructure_decreasing_fact_SpinBox_file")
        self.grid_file.addWidget(self.substructure_decreasing_fact_SpinBox_file, 3, 1, 1, 1)

        self.substructure_decreasing_fact_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_4)
        self.substructure_decreasing_fact_checkBox_file.setText("")
        self.substructure_decreasing_fact_checkBox_file.setObjectName("substructure_decreasing_fact_checkBox_file")
        self.grid_file.addWidget(self.substructure_decreasing_fact_checkBox_file, 3, 2, 1, 1)

        self.label_min_block_thickness_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_min_block_thickness_file.setObjectName("label_min_block_thickness_file")
        self.grid_file.addWidget(self.label_min_block_thickness_file, 4, 0, 1, 1)

        self.substructure_min_block_thickness_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_4)
        self.substructure_min_block_thickness_SpinBox_file.setEnabled(False)
        self.substructure_min_block_thickness_SpinBox_file.setObjectName("substructure_min_block_thickness_SpinBox_file")
        self.grid_file.addWidget(self.substructure_min_block_thickness_SpinBox_file, 4, 1, 1, 1)

        self.substructure_min_block_thickness_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_4)
        self.substructure_min_block_thickness_checkBox_file.setText("")
        self.substructure_min_block_thickness_checkBox_file.setObjectName("substructure_min_block_thickness_checkBox_file")
        self.grid_file.addWidget(self.substructure_min_block_thickness_checkBox_file, 4, 2, 1, 1)

        self.label_max_block_thickness_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_max_block_thickness_file.setObjectName("label_max_block_thickness_file")
        self.grid_file.addWidget(self.label_max_block_thickness_file, 5, 0, 1, 1)

        self.substructure_max_block_thickness_SpinBox_file = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_4)
        self.substructure_max_block_thickness_SpinBox_file.setEnabled(False)
        self.substructure_max_block_thickness_SpinBox_file.setObjectName("substructure_max_block_thickness_SpinBox_file")
        self.grid_file.addWidget(self.substructure_max_block_thickness_SpinBox_file, 5, 1, 1, 1)

        self.substructure_max_block_thickness_checkBox_file = QtWidgets.QCheckBox(self.gridLayoutWidget_4)
        self.substructure_max_block_thickness_checkBox_file.setText("")
        self.substructure_max_block_thickness_checkBox_file.setObjectName("substructure_max_block_thickness_checkBox_file")
        self.grid_file.addWidget(self.substructure_max_block_thickness_checkBox_file, 5, 2, 1, 1)

        self.label_save_result_file = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_save_result_file.setObjectName("label_save_result_file")
        self.grid_file.addWidget(self.label_save_result_file, 6, 0, 1, 1)

        self.substructure_save_result_lineEdit_file = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.substructure_save_result_lineEdit_file.setEnabled(False)
        self.substructure_save_result_lineEdit_file.setObjectName("substructure_save_result_lineEdit_file")
        self.grid_file.addWidget(self.substructure_save_result_lineEdit_file, 6, 1, 1, 1)

        # Substructure Feature Tab -> User Tab
        self.user_tab = QtWidgets.QWidget()
        self.user_tab.setObjectName("user_tab")
        self.tabWidget.addTab(self.user_tab, "")

        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.user_tab)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(0, 3, 421, 231))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")

        self.grid_user = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.grid_user.setContentsMargins(0, 0, 0, 0)
        self.grid_user.setObjectName("grid_user")

        self.label_packet_eq_d_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_packet_eq_d_user.setObjectName("label_packet_eq_d_user")
        self.grid_user.addWidget(self.label_packet_eq_d_user, 0, 0, 1, 1)

        self.substructure_packet_eq_d_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_packet_eq_d_SpinBox_user.setEnabled(True)
        self.substructure_packet_eq_d_SpinBox_user.setObjectName("substructure_packet_eq_d_SpinBox_user")
        self.grid_user.addWidget(self.substructure_packet_eq_d_SpinBox_user, 0, 1, 1, 1)

        self.label_pack_size_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_pack_size_user.setObjectName("label_pack_size_user")
        self.grid_user.addWidget(self.label_pack_size_user, 1, 0, 1, 1)

        self.substructure_pack_size_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_pack_size_SpinBox_user.setEnabled(False)
        self.substructure_pack_size_SpinBox_user.setObjectName("substructure_pack_size_SpinBox_user")
        self.grid_user.addWidget(self.substructure_pack_size_SpinBox_user, 1, 1, 1, 1)

        self.substructure_packet_size_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_packet_size_checkBox_user.setText("")
        self.substructure_packet_size_checkBox_user.setObjectName("substructure_packet_size_checkBox_user")
        self.grid_user.addWidget(self.substructure_packet_size_checkBox_user, 1, 2, 1, 1)

        self.label_circularity_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_circularity_user.setObjectName("label_circularity_user")
        self.grid_user.addWidget(self.label_circularity_user, 2, 0, 1, 1)

        self.substructure_circularity_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_circularity_SpinBox_user.setEnabled(False)
        self.substructure_circularity_SpinBox_user.setObjectName("substructure_circularity_SpinBox_user")
        self.grid_user.addWidget(self.substructure_circularity_SpinBox_user, 2, 1, 1, 1)

        self.substructure_packet_size_file_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_5) # TODO @MAX: falsche Position/Benennung file&user!?
        self.substructure_packet_size_file_checkBox_user.setText("")
        self.substructure_packet_size_file_checkBox_user.setObjectName("substructure_packet_size_file_checkBox_user")
        self.grid_user.addWidget(self.substructure_packet_size_file_checkBox_user, 2, 2, 1, 1)

        self.label_block_thickness_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_block_thickness_user.setObjectName("label_block_thickness_user")
        self.grid_user.addWidget(self.label_block_thickness_user, 3, 0, 1, 1)

        self.substructure_block_thickness_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_block_thickness_SpinBox_user.setEnabled(True)
        self.substructure_block_thickness_SpinBox_user.setObjectName("substructure_block_thickness_SpinBox_user")
        self.grid_user.addWidget(self.substructure_block_thickness_SpinBox_user, 3, 1, 1, 1)

        self.label_decreasing_fact_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_decreasing_fact_user.setObjectName("label_decreasing_fact_user")
        self.grid_user.addWidget(self.label_decreasing_fact_user, 4, 0, 1, 1)

        self.substructure_decreasing_fact_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_decreasing_fact_SpinBox_user.setEnabled(False)
        self.substructure_decreasing_fact_SpinBox_user.setObjectName("substructure_decreasing_fact_SpinBox_user")
        self.grid_user.addWidget(self.substructure_decreasing_fact_SpinBox_user, 4, 1, 1, 1)

        self.substructure_decreasing_fact_file_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_decreasing_fact_file_checkBox_user.setText("")
        self.substructure_decreasing_fact_file_checkBox_user.setObjectName("substructure_decreasing_fact_file_checkBox_user")
        self.grid_user.addWidget(self.substructure_decreasing_fact_file_checkBox_user, 4, 2, 1, 1)

        self.label_min_block_thickness_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_min_block_thickness_user.setObjectName("label_min_block_thickness_user")
        self.grid_user.addWidget(self.label_min_block_thickness_user, 5, 0, 1, 1)

        self.substructure_min_block_thickness_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_min_block_thickness_SpinBox_user.setEnabled(False)
        self.substructure_min_block_thickness_SpinBox_user.setObjectName("substructure_min_block_thickness_SpinBox_user")
        self.grid_user.addWidget(self.substructure_min_block_thickness_SpinBox_user, 5, 1, 1, 1)

        self.substructure_min_block_thickness_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_min_block_thickness_checkBox_user.setText("")
        self.substructure_min_block_thickness_checkBox_user.setObjectName("substructure_min_block_thickness_checkBox_user")
        self.grid_user.addWidget(self.substructure_min_block_thickness_checkBox_user, 5, 2, 1, 1)

        self.label_max_block_thickness_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_max_block_thickness_user.setObjectName("label_max_block_thickness_user")
        self.grid_user.addWidget(self.label_max_block_thickness_user, 6, 0, 1, 1)

        self.substructure_max_block_thickness_SpinBox_user = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_5)
        self.substructure_max_block_thickness_SpinBox_user.setEnabled(False)
        self.substructure_max_block_thickness_SpinBox_user.setObjectName("substructure_max_block_thickness_SpinBox_user")
        self.grid_user.addWidget(self.substructure_max_block_thickness_SpinBox_user, 6, 1, 1, 1)

        self.substructure_max_block_thickness_checkBox_user = QtWidgets.QCheckBox(self.gridLayoutWidget_5)
        self.substructure_max_block_thickness_checkBox_user.setText("")
        self.substructure_max_block_thickness_checkBox_user.setObjectName("substructure_max_block_thickness_checkBox_user")
        self.grid_user.addWidget(self.substructure_max_block_thickness_checkBox_user, 6, 2, 1, 1)

        self.label_save_result_user = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.label_save_result_user.setObjectName("label_save_result_user")
        self.grid_user.addWidget(self.label_save_result_user, 7, 0, 1, 1)

        self.substructure_save_result_lineEdit_user = QtWidgets.QLineEdit(self.gridLayoutWidget_5)
        self.substructure_save_result_lineEdit_user.setEnabled(False)
        self.substructure_save_result_lineEdit_user.setObjectName("substructure_save_result_lineEdit_user")
        self.grid_user.addWidget(self.substructure_save_result_lineEdit_user, 7, 1, 1, 1)

        # Status Tab
        self.status_tab = QtWidgets.QWidget()
        self.InfoPages.addTab(self.status_tab, "")
        self.status_tab.setObjectName("status_tab")

        self.textBrowser = QtWidgets.QTextBrowser(self.status_tab)
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 791, 291))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setText(" Please enter the required information")

        self.verticalScrollBar = QtWidgets.QScrollBar(self.status_tab)
        self.verticalScrollBar.setGeometry(QtCore.QRect(790, 0, 20, 241))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")

        # Log Tab
        self.log_tab = QtWidgets.QWidget()
        self.InfoPages.addTab(self.log_tab, "")
        self.log_tab.setObjectName("log_tab")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents) # TODO @Max: oder nach .addTab log_tab ???

        self.LogFile = QtWidgets.QTextBrowser(self.log_tab)
        self.LogFile.setGeometry(QtCore.QRect(0, 0, 401, 291))
        self.LogFile.setObjectName("LogFile")

        self.Visualization = QtWidgets.QGraphicsView(self.log_tab)
        self.Visualization.setGeometry(QtCore.QRect(400, 0, 321, 291))
        self.Visualization.setObjectName("Visualization")

        ### Main Window
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 340, 320, 101))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.frame_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 10, 311, 91))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setHorizontalSpacing(5)
        self.gridLayout_2.setVerticalSpacing(10)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.frame_1 = QtWidgets.QFrame(self.centralwidget)
        self.frame_1.setGeometry(QtCore.QRect(10, 10, 721, 341))
        self.frame_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_1.setObjectName("frame_1")
        self.formLayoutWidget = QtWidgets.QWidget(self.frame_1)
        self.formLayoutWidget.setGeometry(QtCore.QRect(0, 0, 711, 331))
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
        self.box_sizeSpinBox.setObjectName("box_sizeSpinBox")
        self.gridLayout.addWidget(self.box_sizeSpinBox, 1, 1, 1, 2)
        self.box_sizeSpinBox.valueChanged.connect(self.bandwidth_handler)

        # Resolution:
        self.resolutionLabel = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resolutionLabel.sizePolicy().hasHeightForWidth())
        self.resolutionLabel.setSizePolicy(sizePolicy)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.gridLayout.addWidget(self.resolutionLabel, 2, 0, 1, 1)

        self.resolutionSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget) #TODO: values?
        self.resolutionSpinBox.setDecimals(3)
        #self.resolutionSpinBox.setMinimum()
        self.resolutionSpinBox.setMaximum(10.0)
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
        self.ferriteSpinBox.setSingleStep(0.01)
        self.ferriteSpinBox.setEnabled(False)
        self.ferriteSpinBox.setObjectName("ferriteSpinBox")
        self.gridLayout.addWidget(self.ferriteSpinBox, 5, 1, 1, 2)

        self.martensiteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.martensiteSpinBox.setDecimals(2)
        self.martensiteSpinBox.setMaximum(1.0)
        self.martensiteSpinBox.setSingleStep(0.01)
        self.martensiteSpinBox.setEnabled(False)
        self.martensiteSpinBox.setObjectName("martensiteSpinBox")
        self.gridLayout.addWidget(self.martensiteSpinBox, 5, 4, 1, 1)

        self.pearliteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.pearliteSpinBox.setDecimals(2)
        self.pearliteSpinBox.setMaximum(1.0)
        self.pearliteSpinBox.setSingleStep(0.01)
        self.pearliteSpinBox.setEnabled(False)
        self.pearliteSpinBox.setObjectName("pearliteSpinBox")
        self.gridLayout.addWidget(self.pearliteSpinBox, 5, 6, 1, 1)

        self.bainiteSpinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.bainiteSpinBox.setDecimals(2)
        self.bainiteSpinBox.setMaximum(1.0)
        self.bainiteSpinBox.setSingleStep(0.01)
        self.bainiteSpinBox.setEnabled(False)
        self.bainiteSpinBox.setObjectName("bainiteSpinBox")
        self.gridLayout.addWidget(self.bainiteSpinBox, 5, 8, 1, 1)

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

        self.substructure_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        self.substructure_button.setObjectName("substructure_button")
        self.gridLayout.addWidget(self.substructure_button, 7, 6, 1, 1)
        self.substructure_button.stateChanged.connect(self.features_handler)

        self.roughness_button = QtWidgets.QCheckBox(self.formLayoutWidget)
        #self.roughness_button.setEnabled(False)
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
        self.gridLayout_2.addWidget(self.framework_label, 0, 0, 1, 1)

        self.abaqus_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.abaqus_button.setObjectName("abaqus_button")
        self.gridLayout_2.addWidget(self.abaqus_button, 0, 1, 1, 1)

        self.damask_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.damask_button.setObjectName("damask_button")
        self.gridLayout_2.addWidget(self.damask_button, 0, 2, 1, 1)

        self.moose_button = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.moose_button.setChecked(True)
        self.moose_button.setObjectName("moose_button")
        self.gridLayout_2.addWidget(self.moose_button, 0, 3, 1, 1)

        # Element Type:
        self.element_type_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.element_type_label.setObjectName("element_type_label")
        self.gridLayout_2.addWidget(self.element_type_label, 1, 0, 1, 1)

        self.comboBox_element_type = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox_element_type.setObjectName("comboBox_element_type")
        self.comboBox_element_type.addItem("")
        self.comboBox_element_type.addItem("")
        self.comboBox_element_type.addItem("")
        self.comboBox_element_type.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_element_type, 1, 1, 1, 3)

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
        self.progressBar.setGeometry(QtCore.QRect(10, 790, 281, 23))
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        self.StartButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartButton.setEnabled(False)
        self.StartButton.setGeometry(QtCore.QRect(290, 790, 111, 23))
        self.StartButton.setObjectName("StartButton")

        ###
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionFiles = QtWidgets.QAction(MainWindow)
        self.actionFiles.setObjectName("actionFiles")

        self.retranslateUi(MainWindow)
        self.InfoPages.setCurrentIndex(2)
        self.tabWidget.setCurrentIndex(0)
        self.user_tab.setDisabled(True)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DRAGen - RVE Generator"))
        self.NoBand_label.setText(_translate("MainWindow", "Number of Bands"))
        self.band_thickness_label.setText(_translate("MainWindow", "Band Thickness"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.banding_tab), _translate("MainWindow", "Banding Feature"))
        self.substructure_filemode_radio.setText(_translate("MainWindow", "File Mode"))
        self.substructure_user_mode_radio.setText(_translate("MainWindow", "User Mode"))
        self.label_block_thicknes_file.setText(_translate("MainWindow", "block thickness"))
        self.label_save_result_file.setText(_translate("MainWindow", "save result as:"))
        self.label_pack_size_file.setText(_translate("MainWindow", "Packet size"))
        self.label_max_block_thickness_file.setText(_translate("MainWindow", "maximum blockthickness"))
        self.label_decreasing_fact_file.setText(_translate("MainWindow", "decreasing factor"))
        self.label_min_block_thickness_file.setText(_translate("MainWindow", "minimum blockthickness"))
        self.label_substructure_file.setText(_translate("MainWindow", "Substructure File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.file_tab), _translate("MainWindow", "File"))
        self.label_max_block_thickness_user.setText(_translate("MainWindow", "maximum blockthickness"))
        self.label_decreasing_fact_user.setText(_translate("MainWindow", "decreasing factor"))
        self.label_packet_eq_d_user.setText(_translate("MainWindow", "Packet equiv_d"))
        self.label_min_block_thickness_user.setText(_translate("MainWindow", "minimum blockthickness"))
        self.label_save_result_user.setText(_translate("MainWindow", "save result as:"))
        self.label_block_thickness_user.setText(_translate("MainWindow", "block thickness"))
        self.label_pack_size_user.setText(_translate("MainWindow", "Packet size"))
        self.label_circularity_user.setText(_translate("MainWindow", "circulatrity"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.user_tab), _translate("MainWindow", "User"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.substructure), _translate("MainWindow", "Substructure Feature"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.status_tab), _translate("MainWindow", "Status"))
        self.InfoPages.setTabText(self.InfoPages.indexOf(self.log_tab), _translate("MainWindow", "Log"))
        self.StartButton.setText(_translate("MainWindow", "Start Generation"))
        self.element_type_label.setText(_translate("MainWindow", "Element Type"))
        self.damask_button.setText(_translate("MainWindow", "Damask"))
        self.abaqus_button.setText(_translate("MainWindow", "Abaqus"))
        self.moose_button.setText(_translate("MainWindow", "Moose"))
        self.comboBox_element_type.setItemText(0, _translate("MainWindow", "HEX8 (Moose)"))
        self.comboBox_element_type.setItemText(1, _translate("MainWindow", "C3D4 (Abaqus Tet)"))
        self.comboBox_element_type.setItemText(2, _translate("MainWindow", "C3D10 (Abaqus Tet)"))
        self.comboBox_element_type.setItemText(3, _translate("MainWindow", "C3D8 (Abaqus Hex)"))
        self.framework_label.setText(_translate("MainWindow", "Framework:"))
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

    def bandwidth_handler(self):
        self.band_thicknessSpinBox.setMaximum(self.box_sizeSpinBox.value()/10)

    def button_handler(self):
        if self.MainWindow.sender() == self.fileBrowserFerrite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.p")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditFerrite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserMartensite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.p")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditMartensite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowser_Pearlite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.p")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditPearlite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserBainite:
            dlg = QFileDialog()
            dlg.setNameFilter("*.csv *.p")
            dlg.setFileMode(QFileDialog.AnyFile)
            if dlg.exec_():
                file_path = dlg.selectedFiles()
                self.lineEditBainite.setText(file_path[0])
        elif self.MainWindow.sender() == self.fileBrowserStore_path:
            file_path = QFileDialog.getExistingDirectory()
            self.lineEdit_store_path.setText(file_path)

    def features_handler(self, state):
        if self.MainWindow.sender() == self.Banding_button:
            if state == Qt.Checked:
                self.InfoPages.setCurrentIndex(0)
                self.banding_tab.setEnabled(True)
            else:
                self.banding_tab.setDisabled(True)
        elif self.MainWindow.sender() == self.substructure_button:
            if state == Qt.Checked:
                self.InfoPages.setCurrentIndex(1)
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

    """def save_button_handler(self):
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
                self.twoDcheckBox.setChecked(False)"""

    def submit(self):

        phases = []
        box_size = self.box_sizeSpinBox.value()
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
                self.textBrowser.setText("microstructure imported from:\n{}".format(self.file1))
                if self.file2 is not None:
                    self.textBrowser.setText("and from:\n{}".format(phase1_path))
            elif phase1_path[-4:] == '.csv':
                gan_flag = False
                self.textBrowser.setText("microstructure imported from\n{}".format(phase1_path))
                if phase2_path is not None:
                    self.textBrowser.setText("and from\n{}".format(phase2_path))
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
            self.textBrowser.setText(' Generation running...')
            self.thread.finished.connect(lambda: self.action_submit.setEnabled(True))
            self.thread.finished.connect(lambda: self.textBrowser.setText(' The generation has finished!'))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

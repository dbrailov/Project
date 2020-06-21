# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

import main
from gui import ResultsWindow
# from .ResultsWindow import Ui_ResultsWindow
from gui.ResultsWindow import Ui_ResultsWindow


class EmittingStream(QtCore.QObject):
    def flush(self):
        pass
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class Ui_MainWindow(object):
    def __init__(self, parent=None, **kwargs):
        # Install the custom output stream
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.plainTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.plainTextEdit.setTextCursor(cursor)
        self.plainTextEdit.ensureCursorVisible()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Training")
        MainWindow.resize(1000, 500)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 500))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_train = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_train.setObjectName("pushButton_train")
        self.gridLayout_2.addWidget(self.pushButton_train, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_results = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_results.setObjectName("pushButton_results")
        self.gridLayout_2.addWidget(self.pushButton_results, 3, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_browseAuthor2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_browseAuthor2.setObjectName("pushButton_browseAuthor2")
        self.gridLayout.addWidget(self.pushButton_browseAuthor2, 2, 4, 1, 1)
        self.label_author1 = QtWidgets.QLabel(self.centralwidget)
        self.label_author1.setObjectName("label_author1")
        self.gridLayout.addWidget(self.label_author1, 1, 2, 1, 1)
        self.label_author2 = QtWidgets.QLabel(self.centralwidget)
        self.label_author2.setObjectName("label_author2")
        self.gridLayout.addWidget(self.label_author2, 2, 2, 1, 1)
        self.lineEdit_author1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_author1.setClearButtonEnabled(False)
        self.lineEdit_author1.setObjectName("lineEdit_author1")
        self.gridLayout.addWidget(self.lineEdit_author1, 1, 3, 1, 1)
        self.lineEdit_author2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_author2.setClearButtonEnabled(False)
        self.lineEdit_author2.setObjectName("lineEdit_author2")
        self.gridLayout.addWidget(self.lineEdit_author2, 2, 3, 1, 1)
        self.pushButton_browseAuthor1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_browseAuthor1.setObjectName("pushButton_browseAuthor1")
        self.gridLayout.addWidget(self.pushButton_browseAuthor1, 1, 4, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setPlainText("")
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.gridLayout_2.addWidget(self.plainTextEdit, 0, 1, 3, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_gaussianNoise = QtWidgets.QLabel(self.centralwidget)
        self.label_gaussianNoise.setObjectName("label_gaussianNoise")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_gaussianNoise)
        self.doubleSpinBox_gaussianNoise = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_gaussianNoise.setMaximum(0.1)
        self.doubleSpinBox_gaussianNoise.setSingleStep(0.01)
        self.doubleSpinBox_gaussianNoise.setProperty("value", 0.02)
        self.doubleSpinBox_gaussianNoise.setObjectName("doubleSpinBox_gaussianNoise")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_gaussianNoise)
        self.label_epochs = QtWidgets.QLabel(self.centralwidget)
        self.label_epochs.setObjectName("label_epochs")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_epochs)
        self.spinBox_epochs = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_epochs.setMaximum(100)
        self.spinBox_epochs.setMinimum(2)
        self.spinBox_epochs.setProperty("value", 10)
        self.spinBox_epochs.setObjectName("spinBox_epochs")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox_epochs)
        self.label_filters = QtWidgets.QLabel(self.centralwidget)
        self.label_filters.setObjectName("label_filters")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_filters)
        self.spinBox_filters = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_filters.setMaximum(100)
        self.spinBox_filters.setMinimum(10)
        self.spinBox_filters.setSingleStep(10)
        self.spinBox_filters.setProperty("value", 100)
        self.spinBox_filters.setObjectName("spinBox_filters")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinBox_filters)
        self.label_validation = QtWidgets.QLabel(self.centralwidget)
        self.label_validation.setObjectName("label_validation")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_validation)
        self.doubleSpinBox_validation = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_validation.setMaximum(0.5)
        self.doubleSpinBox_validation.setMinimum(0.1)
        self.doubleSpinBox_validation.setSingleStep(0.05)
        self.doubleSpinBox_validation.setProperty("value", 0.25)
        self.doubleSpinBox_validation.setObjectName("doubleSpinBox_validation")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_validation)
        self.label_pool = QtWidgets.QLabel(self.centralwidget)
        self.label_pool.setObjectName("label_pool")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_pool)
        self.spinBox_pool = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_pool.setMaximum(5)
        self.spinBox_pool.setMinimum(1)
        self.spinBox_pool.setProperty("value", 3)
        self.spinBox_pool.setObjectName("spinBox_pool")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.spinBox_pool)
        self.label_kernel1 = QtWidgets.QLabel(self.centralwidget)
        self.label_kernel1.setObjectName("label_kernel1")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_kernel1)
        self.spinBox_kernel1 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_kernel1.setMaximum(7)
        self.spinBox_kernel1.setProperty("value", 3)
        self.spinBox_kernel1.setObjectName("spinBox_kernel1")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.spinBox_kernel1)
        self.label_kernel2 = QtWidgets.QLabel(self.centralwidget)
        self.label_kernel2.setObjectName("label_kernel2")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_kernel2)
        self.spinBox_kernel2 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_kernel2.setMaximum(7)
        self.spinBox_kernel2.setProperty("value", 4)
        self.spinBox_kernel2.setObjectName("spinBox_kernel2")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.spinBox_kernel2)
        self.label_kernel3 = QtWidgets.QLabel(self.centralwidget)
        self.label_kernel3.setObjectName("label_kernel3")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_kernel3)
        self.spinBox_kernel3 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_kernel3.setMaximum(7)
        self.spinBox_kernel3.setProperty("value", 5)
        self.spinBox_kernel3.setObjectName("spinBox_kernel3")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.spinBox_kernel3)
        self.label_dropout = QtWidgets.QLabel(self.centralwidget)
        self.label_dropout.setObjectName("label_dropout")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_dropout)
        self.doubleSpinBox_dropout = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_dropout.setMaximum(0.5)
        self.doubleSpinBox_dropout.setMinimum(0.1)
        self.doubleSpinBox_dropout.setSingleStep(0.05)
        self.doubleSpinBox_dropout.setProperty("value", 0.5)
        self.doubleSpinBox_dropout.setObjectName("doubleSpinBox_dropout")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_dropout)
        self.gridLayout_2.addLayout(self.formLayout, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 496, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #########################################################################################################
        self.pushButton_browseAuthor1.clicked.connect(self.pushButton_author1_handler)
        self.pushButton_browseAuthor2.clicked.connect(self.pushButton_author2_handler)
        self.pushButton_train.clicked.connect(self.pushButton_train_handler)
        self.pushButton_results.clicked.connect(self.pushButton_results_handler)
        self.pushButton_results.setDisabled(True)

    def pushButton_author1_handler(self):
        # print("pushButton_Author1 pressed!")
        file, value = QFileDialog.getOpenFileName()
        # print(file)
        if file.endswith('.txt'):
            self.lineEdit_author1.setText(file)
        else:
            self.plainTextEdit.appendPlainText("Please enter text file!\n")

    def pushButton_author2_handler(self):
        # print("pushButton_Author2 pressed!")
        file, value = QFileDialog.getOpenFileName()
        # print(file)
        if file.endswith('.txt'):
            self.lineEdit_author2.setText(file)
        else:
            self.plainTextEdit.appendPlainText("Please enter text file!\n")

    def log(self, line):
        self.plainTextEdit.appendPlainText(line)
    # def pushButton_browseEmbedding_handler(self):
    #     print("pushButton_browseEmbedding pressed!")
    #     fileName, value = QFileDialog.getOpenFileName()
    #     print(fileName)
    #     if fileName.endswith('.txt'):
    #         self.lineEdit_embedding.setText(fileName)
    #     else:
    #         self.plainTextEdit.appendPlainText("Must be TXT file")

    def pushButton_train_handler(self):
        self.pushButton_train.setDisabled(True)
        # self.plainTextEdit.appendPlainText("pushButton_train pressed!")
        if (self.lineEdit_author1.text() == '' or self.lineEdit_author2.text() == ''):
            self.plainTextEdit.appendPlainText("Please upload authors text files!\n")
        else:
            main.train(self.lineEdit_author1.text(), self.lineEdit_author2.text(),
                       float(self.doubleSpinBox_gaussianNoise.text()), int(self.spinBox_epochs.text()),
                       int(self.spinBox_filters.text()), float(self.doubleSpinBox_validation.text()),
                       int(self.spinBox_pool.value()), int(self.spinBox_kernel1.value()),
                       int(self.spinBox_kernel2.value()), int(self.spinBox_kernel3.value()),
                       float(self.doubleSpinBox_dropout.text()))
            self.pushButton_results.setDisabled(False)
        self.pushButton_train.setDisabled(False)

    def pushButton_results_handler(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_ResultsWindow()
        self.ui.setupUi(self.window)
        self.window.show()


    #########################################################################################################
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Train", "Train"))
        self.pushButton_train.setText(_translate("Train", "Train"))
        self.label.setText(_translate("Train", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600; font-style:italic;\">Configuration</span></p></body></html>"))
        self.pushButton_results.setText(_translate("Train", "Results"))
        # self.pushButton_browseEmbedding.setText(_translate("MainWindow", "Browse"))
        self.pushButton_browseAuthor2.setText(_translate("Train", "Browse"))
        # self.label_embedding.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Embedding:</span></p></body></html>"))
        self.label_author1.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Author 1:</span></p></body></html>"))
        self.label_author2.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Author 2:</span></p></body></html>"))
        self.pushButton_browseAuthor1.setText(_translate("Train", "Browse"))
        self.label_gaussianNoise.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Gaussian Noise:</span></p></body></html>"))
        self.label_epochs.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Epochs:</span></p></body></html>"))
        self.label_filters.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Filters:</span></p></body></html>"))
        self.label_validation.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Validation Split:</span></p></body></html>"))
        self.label_pool.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Pool Size:</span></p></body></html>"))
        self.label_kernel1.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Kernel Size 1:</span></p></body></html>"))
        self.label_kernel2.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Kernel Size 2:</span></p></body></html>"))
        self.label_kernel3.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Kernel Size 3:</span></p></body></html>"))
        self.label_dropout.setText(_translate("Train", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Dropout:</span></p></body></html>"))
        # self.label_output.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; font-style:italic;\">Output Dim:</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

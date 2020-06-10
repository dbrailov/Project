# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ResultsWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!
import os

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ResultsWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Results")
        MainWindow.resize(412, 391)
        MainWindow.setMinimumSize(QtCore.QSize(412, 391))
        MainWindow.setMaximumSize(QtCore.QSize(412, 391))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(40, 10, 331, 351))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.label_reports = QtWidgets.QLabel(self.splitter)
        self.label_reports.setObjectName("label_reports")
        self.pushButton_kernel3 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_kernel3.setObjectName("pushButton_kernel3")
        self.pushButton_kernel4 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_kernel4.setObjectName("pushButton_kernel4")
        self.pushButton_kernel5 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_kernel5.setObjectName("pushButton_kernel5")
        self.pushButton_accuracyPlot = QtWidgets.QPushButton(self.splitter)
        self.pushButton_accuracyPlot.setObjectName("pushButton_accuracyPlot")
        self.pushButton_lossPlot = QtWidgets.QPushButton(self.splitter)
        self.pushButton_lossPlot.setObjectName("pushButton_lossPlot")
        self.pushButton_varianceCoefficientPlot = QtWidgets.QPushButton(self.splitter)
        self.pushButton_histogramPlot = QtWidgets.QPushButton(self.splitter)
        self.pushButton_varianceCoefficientPlot.setObjectName("pushButton_varianceCoefficientPlot")
        self.pushButton_histogramPlot.setObjectName("pushButton_histogramPlot")
        self.label_conclusions = QtWidgets.QLabel(self.splitter)
        self.label_conclusions.setObjectName("label_conclusions")
        self.pushButton_optimalKernel = QtWidgets.QPushButton(self.splitter)
        self.pushButton_optimalKernel.setObjectName("pushButton_optimalKernel")
        self.pushButton_accuracyPlot.raise_()
        self.label_conclusions.raise_()
        self.pushButton_kernel4.raise_()
        self.pushButton_varianceCoefficientPlot.raise_()
        self.pushButton_histogramPlot.raise_()
        self.pushButton_optimalKernel.raise_()
        self.label_reports.raise_()
        self.pushButton_kernel5.raise_()
        self.pushButton_kernel3.raise_()
        self.pushButton_lossPlot.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 412, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Results", "Results"))
        self.label_reports.setText(_translate("Results", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600; text-decoration: underline; color:#ff0000;\">Reports</span></p></body></html>"))
        self.pushButton_kernel3.setText(_translate("Results", "Kernel 3 Weights"))
        self.pushButton_kernel4.setText(_translate("Results", "Kernel 4 Weights"))
        self.pushButton_kernel5.setText(_translate("Results", "Kernel 5 Weights"))
        self.pushButton_accuracyPlot.setText(_translate("Results", "Accuracy Plot "))
        self.pushButton_lossPlot.setText(_translate("Results", "Loss Plot"))
        self.pushButton_varianceCoefficientPlot.setText(_translate("Results", "Coefficient Of Variance Plot"))
        self.pushButton_histogramPlot.setText(_translate("Results", "Coefficient Of Variance Histogram Plot"))
        self.label_conclusions.setText(_translate("Results", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600; text-decoration: underline; color:#ff0000;\">Conclusions</span></p></body></html>"))
        self.pushButton_optimalKernel.setText(_translate("Results", "Optimal Kernel"))
###################################################################################################
        self.pushButton_kernel3.clicked.connect(self.pushButton_kernel3_handler)
        self.pushButton_kernel4.clicked.connect(self.pushButton_kernel4_handler)
        self.pushButton_kernel5.clicked.connect(self.pushButton_kernel5_handler)
        self.pushButton_accuracyPlot.clicked.connect(self.pushButton_accuracyPlot_handler)
        self.pushButton_lossPlot.clicked.connect(self.pushButton_lossPlot_handler)
        self.pushButton_varianceCoefficientPlot.clicked.connect(self.pushButton_varianceCoefficientPlot_handler)
        self.pushButton_optimalKernel.clicked.connect(self.pushButton_optimalKernel_handler)
        self.pushButton_histogramPlot.clicked.connect(self.pushButton_histogramPlot_handler)
    def pushButton_kernel3_handler(self):
        retval = os.getcwd()
        os.chdir('results/last')
        os.system('2Conv.csv')
        os.chdir(retval)
    def pushButton_kernel4_handler(self):
        retval = os.getcwd()
        os.chdir('results/last')
        os.system('4Conv.csv')
        os.chdir(retval)
    def pushButton_kernel5_handler(self):
        retval = os.getcwd()
        os.chdir('results/last')
        os.system('6Conv.csv')
        os.chdir(retval)
    def pushButton_accuracyPlot_handler(self):
        retval = os.getcwd()
        os.chdir('results/last')
        os.system('AccuracyPlot.png')
        os.chdir(retval)
    def pushButton_lossPlot_handler(self):
        retval = os.getcwd()
        os.chdir('results/last')
        os.system('LossPlot.png')
        os.chdir(retval)
    def pushButton_varianceCoefficientPlot_handler(self):
        retval = os.getcwd()
        os.chdir('results/last')
        os.system('VarianceCoefficientPlot.png')
        os.chdir(retval)
    def pushButton_histogramPlot_handler(self):
        retval = os.getcwd()
        print(retval)
        os.chdir('results/last')
        os.system('HistogramPlot.png')
        os.chdir(retval)
    def pushButton_optimalKernel_handler(self):
        retval = os.getcwd()
        print(retval)
        os.chdir('results/last')
        os.system('OptimalKernel.txt')
        os.chdir(retval)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_ResultsWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

#-*- coding:utf-8 -*-

import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

import mcr_base as mcr

class MainWindow(QtGui.QWidget):
	def __init__(self, parent=None, WfixCheckUse=False, threadUse=False, histUse=None):
		super(MainWindow, self).__init__(parent)
		
		self.__version = '$GITVER$'
		self.setWindowTitle("MCR-ALS")
		
		# option flag
		self.WfixCheckUse = WfixCheckUse
		self.threadUse = threadUse
		self.histUse = histUse
		
		# Widgets
		## buttons
		self.btnLoadA = QtGui.QPushButton('Load 2D data')
		self.btnLoadX = QtGui.QPushButton('Load X axis')
		self.btnInit = QtGui.QPushButton('Init Set')
		self.btnResume = QtGui.QPushButton('Resume')
		self.btnStop = QtGui.QPushButton('Stop')
		self.btnReset = QtGui.QPushButton('Reset')
		self.btnSave = QtGui.QPushButton('Save')
		
		## params
		self.lblRank = QtGui.QLabel('numComp')
		self.spnbxRank = pg.SpinBox(value=4, bounds=[2, None], int=True, step=1)
		self.lblInitMethod = QtGui.QLabel('Init Method')
		self.comboInitMethod = pg.ComboBox(items=['random','SVD','fixed+random'],default='SVD')
		
		self.lblOptMethod = QtGui.QLabel('Opt Method')
		self.comboOptMethod = pg.ComboBox(items=['LS','NNLS'],default='NNLS')
		self.lblCoefL1W = QtGui.QLabel('coef L1W')
		self.spnbxCoefL1W = pg.SpinBox(value=0.0, bounds=[0, None], step=0.001)
		self.lblCoefL2W = QtGui.QLabel('coef L2W')
		self.spnbxCoefL2W = pg.SpinBox(value=0.0, bounds=[0, None], step=0.001)
		self.lblCoefL1H = QtGui.QLabel('coef L1H')
		self.spnbxCoefL1H = pg.SpinBox(value=0.0, bounds=[0, None], step=0.001)
		self.lblCoefL2H = QtGui.QLabel('coef L2H')
		self.spnbxCoefL2H = pg.SpinBox(value=0.0, bounds=[0, None], step=0.001)
		self.lblOptLimit = QtGui.QLabel('opt limit')
		self.spnbxOptLimit = pg.SpinBox(value=0, bounds=[0, None], int=True, step=1)
		
		# create layouts
		layoutBtn = QtGui.QVBoxLayout()
		layoutBtn.addWidget(self.btnLoadA)
		layoutBtn.addWidget(self.btnLoadX)
		
		layoutParam = QtGui.QFormLayout()
		layoutParam.addRow(self.lblRank, self.spnbxRank)
		layoutParam.addRow(self.lblInitMethod, self.comboInitMethod)
		
		layoutBtn2 = QtGui.QVBoxLayout()
		layoutBtn2.addWidget(self.btnInit)
		
		layoutParam2 = QtGui.QFormLayout()
		layoutParam2.addRow(self.lblOptMethod, self.comboOptMethod)
		layoutParam2.addRow(self.lblCoefL1W, self.spnbxCoefL1W)
		layoutParam2.addRow(self.lblCoefL2W, self.spnbxCoefL2W)
		layoutParam2.addRow(self.lblCoefL1H, self.spnbxCoefL1H)
		layoutParam2.addRow(self.lblCoefL2H, self.spnbxCoefL2H)
		layoutParam2.addRow(self.lblOptLimit, self.spnbxOptLimit)
		
		layoutBtn3 = QtGui.QVBoxLayout()
		layoutBtn3sub = QtGui.QHBoxLayout()
		layoutBtn3sub.addWidget(self.btnStop)
		layoutBtn3sub.addWidget(self.btnResume)
		layoutBtn3sub.addWidget(self.btnReset)
		layoutBtn3.addLayout(layoutBtn3sub)
		layoutBtn3.addWidget(self.btnSave)
		
		## optional components
		self.layoutCustom = QtGui.QVBoxLayout()
		self.layoutCustomParam = QtGui.QFormLayout()
		
		if histUse: # if any optional component exist
			lblOption = QtGui.QLabel('---')
			self.layoutCustom.addWidget(lblOption)
		if histUse:
			self.lblHistIntrvl = QtGui.QLabel('hist record intrvl')
			self.spnbxHistIntrvl = pg.SpinBox(value=50, bounds=[1, None], int=True, step=1)
			self.layoutCustomParam.addRow(self.lblHistIntrvl, self.spnbxHistIntrvl)
		
		self.layoutCustom.addLayout(self.layoutCustomParam)
		
		
		# set layouts
		layout = QtGui.QHBoxLayout()
		layoutContainer = QtGui.QVBoxLayout()
		layoutContainer.addLayout(layoutBtn)
		layoutContainer.addLayout(layoutParam)
		layoutContainer.addLayout(layoutBtn2)
		layoutContainer.addLayout(layoutParam2)
		layoutContainer.addLayout(layoutBtn3)
		layoutContainer.addLayout(self.layoutCustom)
		layoutContainer.addStretch(1)
		layout.addLayout(layoutContainer)
		layout.addStretch(1)
		
		self.setLayout(layout)
		
		#signal connect
		self.btnLoadA.clicked.connect(self.onClickedLoadA)
		self.btnLoadX.clicked.connect(self.onClickedLoadX)
		self.btnInit.clicked.connect(self.onClickedInitSet)
		self.btnResume.clicked.connect(self.onClickedStart)
		self.btnStop.clicked.connect(self.onClickedStop)
		self.btnReset.clicked.connect(self.onClickedReset)
		self.btnSave.clicked.connect(self.onClickedSave)
		
		self.spnbxOptLimit.sigValueChanged.connect(self.onChangedOptLimit)
		
		# member
		self.fpathA = ''
		self.fpathX = ''
		self.nmfApp = None
		self.nmfApps = {}
		self.subWs = {}
		self.Wfixs = []
		self.currentIndex = 0
		self.maxIndex = 0
		
		print "MCR-ALS ver. {0}".format(self.__version)
	
	def closeEvent(self, event):
		if len(self.subWs):
			del self.subWs
		event.accept()
	
	def delChild(self, index):
		del self.nmfApps[index]
		del self.subWs[index]
		
		if len(self.subWs.keys()):
			self.currentIndex = max(self.subWs.keys())
		else:
			self.currentIndex = 0
	
	def onClickedLoadA(self):
		filename = pg.FileDialog.getOpenFileName(self, 'open 2D data', filter = "Igor Binary (*.ibw);;all file (*.*)")
		
		if filename:
			fpath = str(filename.toUtf8()).decode('utf-8')
			self.fpathA = fpath
			self.fpathX = ''
			
			print "  >> [mtA] " + self.fpathA
	
	def onClickedLoadX(self):
		filename = pg.FileDialog.getOpenFileName(self, 'open X data', filter = "Igor Binary (*.ibw);;all file (*.*)")
		if filename:
			fpath = str(filename.toUtf8()).decode('utf-8')
			self.fpathX = fpath
			
			print "  >> [Xax] " + self.fpathX
	
	def onClickedInitSet(self):
		if len(self.fpathA):
			
			# create NMFApp
			nmfApp = mcr.NMFApp(self.fpathA)
			if len(self.fpathX):
				nmfApp.load_wvX(self.fpathX)
			
			# create subWindow
			#subW = QtGui.QDialog()
			self.currentIndex = self.maxIndex + 1
			self.maxIndex += 1
			subW = subWindow(self, self.currentIndex)
			subLayout = QtGui.QVBoxLayout()
			
			# initialize NMFApp
			nmfApp['rank'] = self.spnbxRank.value()
			nmfApp['init_method'] = self.comboInitMethod.value()
			initPath = None
			if nmfApp['init_method'] == 'fixed+random':
				filename = pg.FileDialog.getOpenFileName(self, 'open Init wave', filter = "Igor Binary (*.ibw);;all file (*.*)")
				if filename:
					initPath = str(filename.toUtf8()).decode('utf-8')
					print "  >> [ini] " + initPath
				else:
					return
			
			# Wait Cursor
			self.setCursor(QtCore.Qt.WaitCursor)
			
			# initSet
			nmfApp.initSet(initPath)
			
			# embed NMFApp plot
			nmfApp.plotQt_WH(subW)
			subLayoutPlot = QtGui.QHBoxLayout()
			subLayoutPlot.addWidget(nmfApp.glw)
			
			subLayoutInfo = QtGui.QVBoxLayout()
			subLayoutInfo.addWidget(subW.statusText)
			
			# *optional*
			# impleement Ws_fix feature with check box
			if self.WfixCheckUse:
				self.Wfixs = []
				subLayoutCheck = QtGui.QVBoxLayout()
				for r in range(nmfApp['rank']):
					check = QtGui.QCheckBox(str(r+1))
					check.stateChanged.connect(self._Ws_fix_set_array)
					subLayoutCheck.addWidget(check)
					self.Wfixs.append(check)
				subLayoutPlot.addLayout(subLayoutCheck)
			
			# layout
			subLayout.addLayout(subLayoutPlot)
			subLayout.addLayout(subLayoutInfo)
			subW.setLayout(subLayout)
			
			# show subWindow
			subW.show()
			
			# add to list
			self.nmfApps[self.currentIndex] = nmfApp
			self.subWs[self.currentIndex] = subW
			
			#print self.nmfApps
			#print self.subWs
			
			# Cursor reset
			self.unsetCursor()
	
	def onClickedStart(self):
		if not self.nmfApps[self.currentIndex] is None or self.subWs[self.currentIndex] is None:
			self._nmfOptStart()
	
	def onClickedStop(self):
		if self.nmfApps[self.currentIndex] is not None and self.nmfApps[self.currentIndex].Qtimer.isActive():
			self.nmfApps[self.currentIndex].optimize_abort()
	
	def onClickedReset(self):
		if not self.nmfApps[self.currentIndex] is None or self.subWs[self.currentIndex] is None:
			self.nmfApps[self.currentIndex].reset()
			self.nmfApps[self.currentIndex].plotQt_WH(self.subWs[self.currentIndex])
	
	def onClickedSave(self):
		if not self.nmfApps[self.currentIndex] is None or self.subWs[self.currentIndex] is None:
			filename = pg.FileDialog.getSaveFileName(self, 'Save results', filter="Igor binary files (*.ibw)")
			if filename:
				fpath = str(filename.toUtf8()).decode('utf-8')
				self.nmfApps[self.currentIndex].save_file_allResults(fpath)
	
	def onChangedOptLimit(self, sb):
		if not self.nmfApps[self.currentIndex] is None or self.subWs[self.currentIndex] is None:
			self._nmfOptStart()
	
	def _nmfOptStart(self):
		nmfApp = self.nmfApps[self.currentIndex]
		nmfApp.set_opt_limit(self.spnbxOptLimit.value())
		nmfApp.set('coef_L1W', self.spnbxCoefL1W.value())
		nmfApp.set('coef_L2W', self.spnbxCoefL2W.value())
		nmfApp.set('coef_L1H', self.spnbxCoefL1H.value())
		nmfApp.set('coef_L2H', self.spnbxCoefL2H.value())
		
		if self.histUse:
			nmfApp.set('hist_intrvl', self.spnbxHistIntrvl.value())
		
		subW = self.subWs[self.currentIndex]
		
		print ' ** '  + str(subW.windowTitle().toUtf8()) + ' **'
		infostr = "numComp\t%d;\tInit\t%s;\tcycLim\t%d;\talsMet\t%s;\rL1W\t%f;\tL2W\t%f;\tL1H\t%f;\tL2H\t%f;\rWs_fixed\t%s;\trawData\t%s;\tinitData\t%s" % (nmfApp['rank'], nmfApp['init_method'], nmfApp['opt_limit'], nmfApp['opt_method'], nmfApp['coef_L1W'], nmfApp['coef_L2W'], nmfApp['coef_L1H'], nmfApp['coef_L2H'], nmfApp._get_Ws_fix_info(), nmfApp['rawWvName'], nmfApp['initWvName'])
		self.subWs[self.currentIndex].statusText.setPlainText(infostr)
		if self.threadUse:
			nmfApp.optimize_threadQt(parentWin=self.subWs[self.currentIndex])
		else:
			nmfApp.optimize_Qt(parentWin=self.subWs[self.currentIndex])
	
	def _Ws_fix_set_array(self):
		rank = self.nmfApps[self.currentIndex]['rank']
		if len(self.Wfixs) == rank:
			blarr = np.zeros(rank, dtype=bool)
			for i in range(rank):
				if self.Wfixs[i].isChecked():
					blarr[i]=True
			self.nmfApps[self.currentIndex]['Ws_fixed'] = blarr

class subWindow(QtGui.QWidget):
	def __init__(self, parent=None, subWinIndex=0):
		super(subWindow, self).__init__()
		
		self.setWindowTitle("MCR-ALS window{0:d}".format(subWinIndex))
		self.resize(900,600)
		self.mainWin = parent
		self.subWinIndex = subWinIndex
		
		self.statusText = QtGui.QTextEdit("mcr analysis")
		self.statusText.setMaximumHeight(50)
	
	def closeEvent(self, event):
		self.mainWin.delChild(self.subWinIndex)


if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	
	wf = True
	th = False
	hi = False
	if len(sys.argv) > 1:
		if "-wf" in sys.argv:
			# option flag -wf implements Ws_fix checkbox
			wf = True
		if "-th" in sys.argv:
			# option flag -th implements multi-thread calculation
			th = True
		if "-hi" in sys.argv:
			# option flag -hi implements hist_inteval spinBox
			hi = True
	
	main = MainWindow(WfixCheckUse=wf, threadUse=th, histUse=hi)
	main.show()

	sys.exit(app.exec_())

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import datetime
import time

import numpy
import matplotlib.pyplot as plt
from matplotlib import gridspec

import threading

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from ibw import ibw
from solver import als
from solver import nmfsvd

#import pages

class NMFApp(object):
	def __getitem__(self, key):
		return self._params[key]

	def get(self, key, default = None):
		return self._params.get(key, default)

	def __setitem__(self, key, value):
		self._params[key] = value
		#logging.info(['app', key, value])

	def set(self, key, value):
		self[key] = value

	def __init__(self, path = None):
		self.__version = '$GITVER$'
		self._params = {
			'path': None,
			"transposed": False,
			'opt_residue': [],
			'opt_cur': 0,
			'opt_limit': 100,
			'quant_intercept': [],

			'coef_L1W': 0,
			'coef_L2W': 0,
			'coef_L1H': 0,
			'coef_L2H': 0,
			'init_method': 'SVD',
			'opt_method' : 'NNLS',
			'seed' : 0,
			'Ws_fixed': 0,
			'hist_intrvl': 50,
			'rawWvName': "",
			'initWvName': "",
			'qnt_IntercFac': 0,
			'imgNumX': 0,   #for img Disp
			'imgDimSize':None,
			'note':'',
			'started': False,
			}
		
		self.lock = threading.Lock()
		
		if not path:
			filename = pg.FileDialog.getOpenFileName(self, 'open 2D data', filter = "Igor Binary (*.ibw);;all file (*.*)")
			if filename:
				path = str(filename.toUtf8()).decode('utf-8')
				print path
		
		if path:
			self['path'] = os.path.abspath(path)
			self['A'] = numpy.copy(self._load_file(path))
			fname = os.path.basename(path)
			self['rawWvName'] = fname[0:fname.rfind('.ibw')]
			self['wvX']=numpy.arange((numpy.shape(self['A']))[0])
		self['rank'] = 4
		
		self.qgCs = []
		self.qgIms = []
		self.qgRes = []
		self.qgApp = pg.mkQApp()
	
	def _load_file(self, path):
		if not path:
			return None
		wave = ibw.IgorWave()
		wave.load_file(open(path, "rb"))
		if not (wave and wave.validate()) :
			return None
		A = wave.to_numpy_array(numpy)
		logging.info(['loaded ibw:', path, A.shape, A.max(), A.min()])
		if len(wave.note):
			self['note'] = wave.note
			bgI = self['note'].find('imgNumXY:') + 9
			enI = self['note'].find(';')
			if bgI > 8 and enI > -1:
				dimstrs = (self['note'])[bgI:enI].split(':')
				dimlist = []
				for dimstr in dimstrs:
					if len(dimstr):
						dimXY=[]
						for dimXYstr in dimstr.split(','):
							dimXY.append(int(dimXYstr))
						dimlist.append(dimXY)
				if len(dimlist):
					self['imgDimSize']=numpy.array(dimlist)
		
		return A
	
	def load_wvX(self, path = None):
		if not path:
			filename = pg.FileDialog.getOpenFileName(self, 'open X data', filter = "Igor Binary (*.ibw);;all file (*.*)")
			if filename:
				path = str(filename.toUtf8()).decode('utf-8')
		
		if path:
			self['wvX'] = numpy.copy(self._load_file(os.path.abspath(path)))
			if (self['wvX'].shape)[1] == 1:
				self['wvX'] = self['wvX'][:,0]
		
	def set_opt_limit(self, optlim):
		self['opt_limit'] = optlim
		
		if len(self['opt_residue']) > 0:
			res_old = numpy.copy(self['opt_residue'])
			self['opt_residue'] = numpy.empty(self['opt_limit']+1) * numpy.nan
			res_len = min(len(self['opt_residue']), len(res_old))
			self['opt_residue'][:res_len] = res_old[:res_len]
			
		else:
			self['opt_residue'] = numpy.empty(self['opt_limit']+1) * numpy.nan
		
		try:
			self.pltR[0][0].set_xdata(range(optlim+1))
		except:
			pass

	def optimize_console(self, optlim):
		strtT = time.time()
		
		#self.reset()
		print "rank: %d" % self['rank']
		print "init_method: %s" % self['init_method']
		print "opt_limit: %d" % self['opt_limit']
		print "opt_method: %s" % self['opt_method']
		print "coef_L1W: %f" % self['coef_L1W']
		print "coef_L2W: %f" % self['coef_L2W']
		print "coef_L1H: %f" % self['coef_L1H']
		print "coef_L2H: %f" % self['coef_L2H']
		print "hist_intrvl: %d" % self['hist_intrvl']
		print "Ws_fixed: %s" % self._get_Ws_fix_info()
		print "rawWvName: %s" % self['rawWvName']
		print "initWvName: %s" % self['initWvName']
		
		while self['opt_cur'] < self['opt_limit']:
			A = self.get('A', None)
			if not A is None:
				if self.get('transposed', None):
					A = A.T
				try:
					W = self.get('Ws', None)
					H = self.get('Hs', None)
					
					fix_W = self['Ws_fixed']
					logging.debug(["with fixed W", fix_W])
					
					W, H, norms = als.als(A,
										  self.get('Ws', None),
										  self.get('Hs', None),
										  normalize_W = True,
										  coef_L1W = self['coef_L1W'],
										  coef_L2W = self['coef_L2W'],
										  coef_L1H = self['coef_L1H'],
										  coef_L2H = self['coef_L2H'],
										  method = self['opt_method'],
										  fix_W = fix_W)
				except numpy.linalg.linalg.LinAlgError:
					pass
				self['Ws'] = W
				self['Hs'] = H
				self['opt_cur'] += 1
				self['opt_residue'][self['opt_cur']] = norms[0]
				
				#追加
				if self['opt_cur'] % self['hist_intrvl'] == 0:
					self['Whist'] = numpy.dstack((self['Whist'],W))
					self['Hhist'] = numpy.dstack((self['Hhist'],H))
					print self['opt_cur'],
		
		print 
		print 'itr time:',
		print (time.time()-strtT)

	def optimize_sub(self):
		
		#strt_time=datetime.datetime.now()
		#print strt_time
		A = self.get('A', None)
		if not A is None:
			if self.get('transposed', None):
				A = A.T
		
		while self['opt_cur'] < self['opt_limit'] and self['started']:
			try:
				with self.lock:
					W = self.get('Ws', None)
					H = self.get('Hs', None)
				
				fix_W = self['Ws_fixed']
				logging.debug(["with fixed W", fix_W])
				
				W, H, norms = als.als(A,
									  self.get('Ws', None),
									  self.get('Hs', None),
									  normalize_W = True,
									  coef_L1W = self['coef_L1W'],
									  coef_L2W = self['coef_L2W'],
									  coef_L1H = self['coef_L1H'],
									  coef_L2H = self['coef_L2H'],
									  method = self['opt_method'],
									  fix_W = fix_W)
			except numpy.linalg.linalg.LinAlgError:
				print "linalg error"
			
			with self.lock:
				self['Ws'] = W
				self['Hs'] = H
				self['opt_cur'] += 1
				self['opt_residue'][self['opt_cur']] = norms[0]
				
				if self['opt_cur'] % self['hist_intrvl'] == 0:
					self['Whist'] = numpy.dstack((self['Whist'],W))
					self['Hhist'] = numpy.dstack((self['Hhist'],H))
		
		self['started'] = False
	
	def optimize_thread(self, pltON=True, parentWin=None, closePlot=False):
		if pltON:
			self.plot_WH()
		
		strtT = time.time()

		if parentWin is None:
			print "rank: %d" % self['rank']
			print "init_method: %s" % self['init_method']
			print "opt_limit: %d" % self['opt_limit']
			print "opt_method: %s" % self['opt_method']
			print "coef_L1W: %f" % self['coef_L1W']
			print "coef_L2W: %f" % self['coef_L2W']
			print "coef_L1H: %f" % self['coef_L1H']
			print "coef_L2H: %f" % self['coef_L2H']
			print "hist_intrvl: %d" % self['hist_intrvl']
			print "Ws_fixed: %s" % self._get_Ws_fix_info()
			print "rawWvName: %s" % self['rawWvName']
			print "initWvName: %s" % self['initWvName']
		
		p = threading.Thread(target=self.optimize_sub)
		self['started'] = True
		pltIntrvl = self['pltIntrvl']/1000.0
		p.start()
		
		while self['started']:
			if pltON:
				self.update_plot_WH()
				plt.pause(pltIntrvl)
			else:
				time.sleep(1)
			
			print self['opt_cur'],
		
		print 
		print 'itr time:',
		print (time.time()-strtT)
		
		if pltON:
			self.update_plot_WH()
			plt.pause(pltIntrvl)
			
			if closePlot:
				plt.close()
	
	def optimize_threadQt(self, pltON=True, parentWin=None, closePlot=False):
		self.strtT = time.time()
		
		if parentWin is None:
			print "rank: %d" % self['rank']
			print "init_method: %s" % self['init_method']
			print "opt_limit: %d" % self['opt_limit']
			print "opt_method: %s" % self['opt_method']
			print "coef_L1W: %f" % self['coef_L1W']
			print "coef_L2W: %f" % self['coef_L2W']
			print "coef_L1H: %f" % self['coef_L1H']
			print "coef_L2H: %f" % self['coef_L2H']
			print "hist_intrvl: %d" % self['hist_intrvl']
			print "Ws_fixed: %s" % self._get_Ws_fix_info()
			print "rawWvName: %s" % self['rawWvName']
			print "initWvName: %s" % self['initWvName']
		
		strt_time=datetime.datetime.now()
		print " ..optimize_threadQt start ",
		print strt_time
		
		p = threading.Thread(target=self.optimize_sub)
		self['started'] = True
		p.start()
		
		self.cur_prev = self['opt_cur']
		self.cur_init = self['opt_cur']
		self.plotQt_WH(parentWin)
		self.Qtimer = QtCore.QTimer()
		self.Qtimer.timeout.connect(self.update_thread_plotWH)
		self.Qtimer.start(70)
		
		if pltON and closePlot:
			pass
	
	def optimize_Qt(self, parentWin=None):
		self.strtT = time.time()
		
		#qtApp = QtGui.QApplication([])
		self.plotQt_WH(parentWin)
		
		#self.reset()
		
		if parentWin is None:
			print "rank: %d" % self['rank']
			print "init_method: %s" % self['init_method']
			print "opt_limit: %d" % self['opt_limit']
			print "opt_method: %s" % self['opt_method']
			print "coef_L1W: %f" % self['coef_L1W']
			print "coef_L2W: %f" % self['coef_L2W']
			print "coef_L1H: %f" % self['coef_L1H']
			print "coef_L2H: %f" % self['coef_L2H']
			print "hist_intrvl: %d" % self['hist_intrvl']
			print "Ws_fixed: %s" % self._get_Ws_fix_info()
			print "rawWvName: %s" % self['rawWvName']
			print "initWvName: %s" % self['initWvName']
		
		strt_time=datetime.datetime.now()
		print " ..optimization start ",
		print strt_time
		
		A = self.get('A', None)
		if not A is None:
			if self.get('transposed', None):
				A = A.T
		
		self.cur_init = self['opt_cur']
		self['started'] = True
		self.Qtimer = QtCore.QTimer()
		self.Qtimer.timeout.connect(self.QtUpdateData)
		self.Qtimer.start()
		
		#qtApp.exec_()
		
	def QtUpdateData(self):
		A = self.get('A', None)
		if not A is None:
			if self.get('transposed', None):
				A = A.T
		
		if self['opt_cur'] < self['opt_limit'] and self['started'] and self.glw.isVisible():
			try:
				W = self.get('Ws', None)
				H = self.get('Hs', None)
				
				fix_W = self['Ws_fixed']
				logging.debug(["with fixed W", fix_W])
				
				W, H, norms = als.als(A,
									  self.get('Ws', None),
									  self.get('Hs', None),
									  normalize_W = True,
									  coef_L1W = self['coef_L1W'],
									  coef_L2W = self['coef_L2W'],
									  coef_L1H = self['coef_L1H'],
									  coef_L2H = self['coef_L2H'],
									  method = self['opt_method'],
									  fix_W = fix_W)
			except numpy.linalg.linalg.LinAlgError:
				print "linalg error"
			
			self['Ws'] = W
			self['Hs'] = H
			self['opt_cur'] += 1
			self['opt_residue'][self['opt_cur']] = norms[0]
			self['errSpc'] = norms[1]
			
			self.update_plotQt_WH()
			
			if self['opt_cur'] % self['hist_intrvl'] == 0:
				self['Whist'] = numpy.dstack((self['Whist'],W))
				self['Hhist'] = numpy.dstack((self['Hhist'],H))
		else:
			self['started'] = False
			if self['opt_cur'] == self['opt_limit']:
				print ' ..completed',
			else:
				print ' ..interupted',
			print ".. {0:d} cycle, {1:f} sec.".format((self['opt_cur']-self.cur_init), (time.time()-self.strtT))
			print ' ..itr time:',
			print (time.time()-self.strtT)
			if self.Qtimer.isActive():
				self.Qtimer.stop()
	
	def plotQt_WH(self, parent=None):
		rank = self['rank']
		
		if not self['imgDimSize'] is None:
			self['imgNumX'] = max(self['imgDimSize'][:,0])
			self['numImg'] = self['imgDimSize'].shape[0]
		
		if len(self.qgCs) != rank:
			
			wvX = self['wvX']
			Ws = self['Ws']
			Hs = self['Hs']
			Res = self['opt_residue']
			errSpc = self['errSpc']
			
			self.glw = pg.GraphicsLayoutWidget()
			self.qgRes = []
			self.qgCs = []
			self.qgIms = []
			pos = numpy.array([0.0, 0.5, 1.0])
			
			# plot Res
			cr = pg.PlotCurveItem()
			cr.setData(Res)
			self.glw.addPlot(row=0, col=0, rowspan=(rank-1)).addItem(cr)
			self.qgRes.append(cr)
			
			cr = pg.PlotCurveItem()
			cr.setData(wvX, errSpc)
			self.glw.addPlot(row=(rank-1), col=0).addItem(cr)
			self.glw.getItem((rank-1), 0).showGrid(True, False, 0.9)
			self.qgRes.append(cr)
			
			for j in range(rank):
				# plot W
				c = pg.PlotCurveItem(pen=(j,rank))
				c.setData(wvX, Ws[:,j])
				self.glw.addPlot(row=j, col=1).addItem(c)
				self.glw.getItem(j, 1).showGrid(True, False, 0.9)
				self.glw.getItem(j, 1).setMinimumWidth(300)
				self.qgCs.append(c)
				
				# plot H
				if(self['imgNumX']==0):
					Himg = numpy.empty((Hs.shape[1],1))
					Himg[:,0] = Hs[j,:]
				else:
					Himg = numpy.zeros((sum(self['imgDimSize'][:,1]),self['imgNumX']))
					imgOffsetY=0
					imgOffset=0
					for k in range(self['numImg']):
						Himgk = numpy.reshape(Hs[j,imgOffset:(imgOffset+self['imgDimSize'][k,0]*self['imgDimSize'][k,1])], (self['imgDimSize'][k,1],self['imgDimSize'][k,0]))
						Himg[imgOffsetY:(imgOffsetY+self['imgDimSize'][k,1]),0:self['imgDimSize'][k,0]] = numpy.copy(Himgk[:,:])
						imgOffsetY += self['imgDimSize'][k,1]
						imgOffset += (self['imgDimSize'][k,0]*self['imgDimSize'][k,1])
				
				im = pg.ImageItem(Himg)
				color = numpy.array([(0,0,0,255), pg.colorTuple(pg.intColor(j,rank)), (255,255,255,255)], dtype=numpy.ubyte)
				map = pg.ColorMap(pos, color)
				lut = map.getLookupTable(0.0, 1.0, 256)
				im.setLookupTable(lut)
				
				self.glw.addViewBox(row=j, col=2).addItem(im)
				self.qgIms.append(im)
			
			self.glw.getItem(0,0).setTitle("cycle: {0:d}".format(self['opt_cur']))
			if parent is None:
				self.glw.setWindowTitle("MCR-ALS optimization")
				self.glw.show()
			
		else:
			self.update_plotQt_WH()
	
	def plot_WH(self, fg=None):
		rank = self['rank']
		if self['transposed']:
			size = (self['A'].shape[1], self['A'].shape[0])
		else:
			size = self['A'].shape
		
		imgNumX = self['imgNumX']
		if(imgNumX==0):
			xH = numpy.arange(size[1])
			self['imgSize'] = None
		else:
			self['imgSize'] = ((size[1]//imgNumX), imgNumX)
		
		if not self['imgDimSize'] is None:
			self['imgNumX'] = max(self['imgDimSize'][:,0])
			self['numImg'] = self['imgDimSize'].shape[0]
			numImg = self['numImg']
			
			widR = max(1, min(4, ((sum(self['imgDimSize'][:,1]))//(max(self['imgDimSize'][:,0])))/2))
		else:
			widR = 2
		
		wvX = self['wvX']
		Ws = self['Ws']
		Hs = self['Hs']
		Res = self['opt_residue']
		
		if fg is None:
			fg = plt.figure()
			fgNew = True
		else:
			fgNew = False
		
		self.pltW=[]
		self.pltH=[]
		self.pltR=[]
		gs = gridspec.GridSpec(rank+1, 2, width_ratios=[(5-widR), widR])
		self.axs=[]
		for j in range(rank):
			#plot W
			self.axs.append(fg.add_subplot(gs[j*2]))
			self.pltW.append(self.axs[j*2].plot(wvX, Ws[:, j]))
			self.axs[j*2].set_xlim(wvX[0], wvX[-1])
			self.axs[j*2].grid(True)
		
			# plot H
			if(self['imgNumX']==0):
				self.axs.append(fg.add_subplot(gs[j*2+1]))
				self.pltH.append(self.axs[j*2+1].plot(xH, Hs[j,:]))
				self.axs[j*2+1].set_xlim(xH[0], xH[-1])
			else:
				Himg = numpy.empty((sum(self['imgDimSize'][:,1]),self['imgNumX'])) * numpy.nan
				imgOffsetY=0
				imgOffset=0
				for k in range(numImg):
					Himgk = numpy.reshape(Hs[j,imgOffset:(imgOffset+self['imgDimSize'][k,0]*self['imgDimSize'][k,1])], (self['imgDimSize'][k,1],self['imgDimSize'][k,0]))
					Himg[imgOffsetY:(imgOffsetY+self['imgDimSize'][k,1]),0:self['imgDimSize'][k,0]] = numpy.copy(Himgk[:,:])
					imgOffsetY += self['imgDimSize'][k,1]
					imgOffset += (self['imgDimSize'][k,0]*self['imgDimSize'][k,1])
					
				self.axs.append(fg.add_subplot(gs[j*2+1]))
				self.axs[j*2+1].imshow(Himg.T, interpolation='none')
		
		# plot res
		#self.axs.append(fg.subplot2grid((rank+1, 2), (rank,0), colspan=2))
		self.axs.append(fg.add_subplot(gs[-1:,:]))
		self.pltR.append(self.axs[rank*2].plot(Res))
		self.axs[rank*2].set_xlim(0, self['opt_limit'])
		self.axs[rank*2].set_ylim(0, self['opt_residue'][0])
		self.axs[rank*2].set_title("{0:d} / {1:d}".format(self['opt_cur'], self['opt_limit']))
		
		if fgNew:
			plt.pause(0.01)
	
	def update_thread_plotWH(self):
		if self.cur_prev < self['opt_cur']:
			if self['opt_cur'] < self['opt_limit'] and self['started']:
				self.update_plotQt_WH()
			else:
				self['started'] = False
				
				self.update_plotQt_WH()
				if self['opt_cur'] == self['opt_limit']:
					print ' ..completed',
				else:
					print ' ..interupted',
				print ".. {0:d} cycle, {1:f} sec.".format((self['opt_cur']-self.cur_init), (time.time()-self.strtT))
				if self.Qtimer.isActive():
					self.Qtimer.stop()
		
		self.cur_prev = self['opt_cur']
	
	def update_plotQt_WH(self):
		#with self.lock:
		wvX = self['wvX']
		Ws = self['Ws']
		Hs = self['Hs']
		Res = self['opt_residue']
		errSpc = self['errSpc']
		rank = self['rank']
		imgDimSize = self.get('imgDimSize', None)
		numImg = self.get('numImg',None)
		imgNumX = self['imgNumX']
		ocr = self['opt_cur']
		olim = self['opt_limit']
		
		# residuals
		self.qgRes[0].setData(Res)
		self.qgRes[1].setData(wvX, errSpc)
		
		for j in range(rank):
			# W plot
			self.qgCs[j].setData(wvX, Ws[:,j])
			
			# H plot
			if imgDimSize is None:
				Himg = numpy.empty((Hs.shape[1],1))
				Himg[:,0] = Hs[j,:]
			else:
				Himg = numpy.zeros((sum(imgDimSize[:,1]), imgNumX))
				imgOffsetY=0
				imgOffset=0
				for k in range(numImg):
					Himgk = numpy.reshape(Hs[j,imgOffset:(imgOffset+imgDimSize[k,0]*imgDimSize[k,1])], (imgDimSize[k,1],imgDimSize[k,0]))
					Himg[imgOffsetY:(imgOffsetY+imgDimSize[k,1]),0:imgDimSize[k,0]] = numpy.copy(Himgk[:,:])
					imgOffsetY += imgDimSize[k,1]
					imgOffset += (imgDimSize[k,0]*imgDimSize[k,1])
			
			self.qgIms[j].setImage(Himg)
		
		self.glw.getItem(0,0).setTitle("cycle: {0:d}".format(self['opt_cur']))
		#self.glw.setWindowTitle("MCR-ALS optimization: {0:d}".format(self['opt_cur']))
		#plt.pause(self['pltIntrvl']/1000)
	
	def update_plot_WH(self):
		#with self.lock:
		Ws = self['Ws']
		Hs = self['Hs']
		Res = self['opt_residue']
		rank = self['rank']
		imgDimSize = self.get('imgDimSize', None)
		numImg = self.get('numImg',None)
		imgNumX = self['imgNumX']
		ocr = self['opt_cur']
		olim = self['opt_limit']
		
		for j in range(rank):
			self.pltW[j][0].set_ydata(Ws[:, j])
			self.axs[j*2].set_ylim(min(Ws[:,j]), max(Ws[:,j]))
			
			if imgDimSize is None:
				self.pltH[j][0].set_ydata(Hs[j,:])
				self.axs[j*2+1].set_ylim(min(Hs[j,:]), max(Hs[j,:]))
			else:
				Himg = numpy.empty((sum(imgDimSize[:,1]), imgNumX)) * numpy.nan
				imgOffsetY=0
				imgOffset=0
				for k in range(numImg):
					Himgk = numpy.reshape(Hs[j,imgOffset:(imgOffset+imgDimSize[k,0]*imgDimSize[k,1])], (imgDimSize[k,1],imgDimSize[k,0]))
					Himg[imgOffsetY:(imgOffsetY+imgDimSize[k,1]),0:imgDimSize[k,0]] = numpy.copy(Himgk[:,:])
					imgOffsetY += imgDimSize[k,1]
					imgOffset += (imgDimSize[k,0]*imgDimSize[k,1])
				
				self.axs[j*2+1].imshow(Himg.T, interpolation='none')
		
		self.pltR[0][0].set_ydata(Res)
		self.axs[rank*2].set_xlim(0, ocr+1)
		self.axs[rank*2].set_ylim(0, max(Res[0:(ocr+1)])*1.05)
		self.axs[rank*2].set_title("{0:d} / {1:d}".format(ocr, olim))
		
		#plt.pause(self['pltIntrvl']/1000)
	
	def run(self):
		pass
	
	def optimize_abort(self):
		self['started'] = False

	def reset(self):
		if not self.get('W', None) is None:
			self['Ws'] = numpy.copy(self['W'])
			self['Whist'] = numpy.copy(self['W'])
		else:
			self['Ws'] = None
			self['Whist'] = None
		if not self.get('H', None) is None:
			self['Hs'] = numpy.copy(self['H'])
			self['Hhist'] = numpy.copy(self['H'])
		else:
			self['Hs'] = None
			self['Hhist'] = None
		
		self['opt_residue'][:] = numpy.nan
		self['opt_cur'] = 0
		
		if not self['Ws'] is None or self['Hs'] or None:
			err_mat = self['A'] - numpy.dot(self['Ws'], self['Hs'])
			self['opt_residue'][0] = numpy.linalg.norm(err_mat)
			self['errSpc'] = numpy.sqrt(numpy.sum(err_mat**2, 1))
		
		#self.qgCs = []
		#self.qgIms = []
		#self.qgRes = []
	
	def initSet(self, path=None):
		if not self['A'] is None:
			self['opt_cur'] = 0
			self['opt_residue'] = numpy.empty(self['opt_limit']+1) * numpy.nan
			self['pltIntrvl'] = 500
			self['Ws_fixed'] = 0
			self.qgCs = []
			self.qgIms = []
			self.qgRes = []
			
			if self['init_method'] == 'SVD':
				sT0 = time.time()
				self['W'], self['H'], s = nmfsvd.guess_by_svd(self['A'], self['rank'])
				self['pltIntrvl'] = int(max(0.1, (time.time() - sT0)) * 1000)
				self['initWvName'] = ""
				self['Ws'] = numpy.copy(self['W'])
				self['Hs'] = numpy.copy(self['H'])
				err_mat = self['A'] - numpy.dot(self['W'], self['H'])
				self['opt_residue'][0] = numpy.linalg.norm(err_mat)
				self['errSpc'] = numpy.sqrt(numpy.sum(err_mat**2, 1))
				
			elif self['init_method'] == 'random':
				rows, cols = self['A'].shape
				rank = self['rank']
				#random floats in the half-open interval [0.0, 1.0)
				seed = int(self['seed'])
				numpy.random.seed(seed)
				W = numpy.random.random_sample( (rows, rank) )
				self['W'] = W
				W, H, norms = als.als(self.get('A',None),
									W,
									None,
									normalize_W = True,
									coef_L1W = self['coef_L1W'],
									coef_L2W = self['coef_L2W'],
									coef_L1H = self['coef_L1H'],
									coef_L2H = self['coef_L2H'],
									method = self['opt_method'],)
				self['H'] = H
				self['Ws'] = numpy.copy(self['W'])
				self['Hs'] = numpy.copy(self['H'])
				err_mat = self['A'] - numpy.dot(self['W'], self['H'])
				self['opt_residue'][0] = numpy.linalg.norm(err_mat)
				self['errSpc'] = numpy.sqrt(numpy.sum(err_mat**2, 1))
				
			elif self['init_method'] == 'fixed+random':
				rows, cols = self['A'].shape
				numpy.random.seed(0)	#seed
				W = numpy.random.random_sample( (rows, self['rank']) )
				
				if not path:
					filename = pg.FileDialog.getOpenFileName(caption = 'open Init wave', filter = "Igor Binary (*.ibw);;all file (*.*)")
					if filename:
						path = str(filename.toUtf8()).decode('utf-8')
				
				if path:
					path = os.path.abspath(path)
					W_fixed = numpy.copy(self._load_file(path))
					# W_fixed -> normalize
					for c in range(W_fixed.shape[1]):
						n = numpy.linalg.norm(W_fixed[:, c])
						if n > 0:
							W_fixed[:, c] /= n
					if len(W_fixed.shape) == 1: 
						for x in range(W_fixed.shape[0]):
							W[x][0] = W_fixed[x]
							self['Ws_fixed'] = 1
					else: # assume to be a 2D wave
						if W.shape[0] == W_fixed.shape[0]:
							n = min(W.shape[1], W_fixed.shape[1])
							for y in range(n):
								for x in range(W_fixed.shape[0]):
									W[x][y] = W_fixed[x][y]
							self['Ws_fixed'] =  y + 1   #self['Ws_fixed'] - num of fixed comp.
						elif W.shape[0] == W_fixed.shape[1]:
							n = min(W.shape[1], W_fixed.shape[0])
							for y in range(n):
								for x in range(W_fixed.shape[1]):
									W[x][y] = W_fixed[y][x]
							self['Ws_fixed'] = y + 1
						else:
							logging.error(['dimension mismatch',
											W.shape, W_fixed.shape])
					self['W'] = W
					W, H, norms = als.als(self.get('A',None),
										W,
										None,
										normalize_W = True,
										coef_L1W = self['coef_L1W'],
										coef_L2W = self['coef_L2W'],
										coef_L1H = self['coef_L1H'],
										coef_L2H = self['coef_L2H'],
										method = self['opt_method'],)
					self['H'] = H
					self['Ws'] = numpy.copy(self['W'])
					self['Hs'] = numpy.copy(self['H'])
					err_mat = self['A'] - numpy.dot(self['W'], self['H'])
					self['opt_residue'][0] = numpy.linalg.norm(err_mat)
					self['errSpc'] = numpy.sqrt(numpy.sum(err_mat**2, 1))
					fname=os.path.basename(path)
					self['initWvName'] = fname[0:fname.rfind('.ibw')]
			
			# hist 保存
			self['Whist'] = numpy.copy(self['W'])
			self['Hhist'] = numpy.copy(self['H'])
			
	
	def save_file_allResults(self, path = None):
		if path == None:
			savepath = pg.FileDialog.getSaveFileName(caption = "Save results", filter = "Igor Binary (*.ibw);;all file (*.*)")
			if savepath:
				path = str(filename.toUtf8()).decode('utf-8')
			else:
				return
		
		if path:
			path = os.path.abspath(path).encode('utf-8')
			file_name = path
			folder_name = os.path.dirname(path)
			fname = os.path.basename(path)
			if fname.rfind('.ibw') == -1:
				basename = fname
			else:
				basename = fname[0:(fname.rfind('.ibw'))]
			
			#print file_name
			#print folder_name
			#print basename
			self._save_ndarray(folder_name, ("W_"+basename), self.get('Ws', None))
			self._save_ndarray(folder_name, ("H_"+basename), self.get('Hs', None), note=self['note'])
			self._save_ndarray(folder_name, ("RES_"+basename), self.get('opt_residue', None))
			self._save_ndarray(folder_name, ("Winit_"+basename), self.get('W', None))
			self._save_ndarray(folder_name, ("Whist_"+basename), self.get('Whist', None))
			self._save_ndarray(folder_name, ("Hhist_"+basename), self.get('Hhist', None))
			
			file_name = os.path.join(folder_name, 'NMFinfo_' + basename + '.txt').decode('utf-8')
			f = open(file_name, "w")
			f.write("basename:%s;numComp:%d;InitMethod:%s;numCycl:%d;alsMethod:%s;L1W:%f;L2W:%f;L1H:%f;L2H:%f;Ws_fixed:%s;rawDataWvName:%s;initDataWvName:%s;histInterval:%d;%s\r" % (basename, self['rank'], self['init_method'], self['opt_cur'], self['opt_method'], self['coef_L1W'], self['coef_L2W'], self['coef_L1H'], self['coef_L2H'], self._get_Ws_fix_info(), self['rawWvName'], self['initWvName'], self['hist_intrvl'], self['note']))
			f.write("Ws and Hs were recorded in every %d cycles." % self['hist_intrvl'])
			f.close()
	
	def save_file_currResults(self, path = None):
		if path == None:
			savepath = pg.FileDialog.getSaveFileName(caption = "Save result", filter = "Igor Binary (*.ibw);;all file (*.*)")
			if savepath:
				path = str(savepath.toUtf8()).decode('utf-8')
			else:
				return
		
		if path:
			path = os.path.abspath(path).encode('utf-8')
			file_name = path
			folder_name = os.path.dirname(path)
			fname = os.path.basename(path)
			if fname.rfind('.ibw') == -1:
				basename = fname
			else:
				basename = fname[0:(fname.rfind('.ibw'))]
			
			#print file_name
			#print folder_name
			#print basename
			self._save_ndarray(folder_name, ("W_"+basename), self.get('Ws', None))
			self._save_ndarray(folder_name, ("H_"+basename), self.get('Hs', None))
			self._save_ndarray(folder_name, ("RES_"+basename), self.get('opt_residue', None))
			#self._save_ndarray(folder_name, ("Whist_"+basename), self.get('Whist', None))
			#self._save_ndarray(folder_name, ("Hhist_"+basename), self.get('Hhist', None))
			
			file_name = os.path.join(folder_name, 'NMFinfo_'+basename+'.txt').decode('utf-8')
			f = open(file_name, "w")
			f.write("basename:%s;numComp:%d;InitMethod:%s;numCycl:%d;alsMethod:%s;L1W:%f;L2W:%f;L1H:%f;L2H:%f;Ws_fixed:%s;rawDataWvName:%s;initDataWvName:%s;histInterval:%d;%s\r" % (basename, self['rank'], self['init_method'], self['opt_cur'], self['opt_method'], self['coef_L1W'], self['coef_L2W'], self['coef_L1H'], self['coef_L2H'], self._get_Ws_fix_info(), self['rawWvName'], self['initWvName'], self['hist_intrvl'], self['note']))
			f.close()

	def _save_ndarray(self, folder_name, wave_name, ndarray, note=''):
		if ndarray is None:
			return

		wave = ibw.IgorWave()
		size = ndarray.shape
		if len(size) == 0:
			size = (0,0,0)
		elif len(size) == 1:
			size = (size[0],0,0)
		elif len(size) == 2:
			size = (size[0],size[1],0)
		to_save = numpy.copy(ndarray)

		to_save = to_save.astype(numpy.float32)
		to_save = numpy.require(to_save, requirements = ['F'])

		blob = to_save.data
		logging.info([size, len(blob)])
		wave.set_blob(wave_name + ("\x00" * (32 - len(wave_name))), size, blob, note=note)
		
		file_name = os.path.join(folder_name, wave_name+'.ibw').decode('utf-8')
		print '  >> [save ibw] ' + file_name
		logging.info(['save ibw:', file_name])
		wave.save_file(open(file_name, 'wb'))

	def _get_Ws_fix_info(self):
		retstr = ""
		if type(self['Ws_fixed']) == numpy.ndarray:
			for bl in self['Ws_fixed']:
				if bl:
					retstr += 'T'
				else:
					retstr += 'F'
		else:
			retstr = '%d' % self['Ws_fixed']
		
		return retstr


def main(args):
	app = NMFApp(*args) 
	app.run()

if __name__ == "__main__":
	logging.basicConfig(level = logging.INFO)
	main(sys.argv[1:])

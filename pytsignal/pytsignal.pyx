
import numpy as np
cimport numpy as np
import pandas as pd
from pandas import DataFrame as df
from datetime import datetime


class pytSignal:
	""" 
	alpha: Smoothing Parameter for Tracking Signal
	method: choose between SMA or EWMA for estimator. Default is SMA
	gamma: Smoothing Parameter for EWMA
	"""
	__ticker = np.array([]) # stock's ticker list
	__snum = 0 # # of stocks	
	__win = 0
	__gamma = 0.0
	__alpha = 0.0 
	__method = 'SMA'	
	__sreturns = df() # stock returns list
	__rerr = df() # rerr data
	__Et = df() # error term data
	__Mt = df() # MAD data
	__price = df()
	__tsignal = df()

	def __init__(self, pdata=df(), window=250, alpha=0.1, method='SMA', gamma=0.1):
		
		self.__price = pdata
		self.__ticker = pdata.columns.values
		self.__snum = pdata.columns.values.size
		self.__win = window
		self.__alpha = alpha
		self.__method = method
		self.__gamma = gamma
		self.calRerr()
		self.caltSignal()

	def __del__(self):
		print("This pytSignal Dies!")

	
	def calRerr(self):
		self.__sreturns = np.log(self.__price / self.__price.shift(1))
		self.__sreturns = self.__sreturns.dropna()
		if self.__method == 'SMA':
			est = self.__sreturns.rolling(window=self.__win).mean()
		elif self.__method == 'EWMA':
			est = self.__sreturns.ewm(alpha=self.__gamma, min_periods=self.__win).mean()

		self.__rerr = self.__sreturns.copy()
		self.__rerr -= est.shift(1)

		self.__rerr = self.__rerr.dropna()


	
	def caltSignal(self):
		self.__Et = self.__rerr.ewm(alpha=self.__alpha).mean()
		reabs = np.abs(self.__rerr)
		self.__Mt = reabs.ewm(alpha=self.__alpha).mean()
		
		self.__tsignal = self.__Et / self.__Mt

	def chData(self, pdata):
		self.__price = pdata
		self.__ticker = pdata.columns.values
		self.__snum = pdata.columns.values.size
		self.calRerr()
		self.caltSignal()

	def getmt(self):
		return self.__Mt

	def getet(self):
		return self.__Et

	def tsignal(self):
		return self.__tsignal

	def getstockn(self):
		return self.__snum

	def getalpha(self):
		return self.__alpha

	def setalpha(self, alpha):
		self.__alpha = alpha

	def getprice(self):
		return self.__price

	def getreturns(self):
		return self.__sreturns

	def getrerr(self):
		return self.__rerr

	def getticker(self):
		return self.__ticker

	def getlast(self):
		return self.__tsignal.ix[-1]

cpdef np.ndarray rarray(np.ndarray values, int p):
	cdef int nm = values.shape[1]
	cdef int length = values.shape[0]
	cdef int rlen

	if length < p:
		raise ValueError("The length of Array is shorter than requested period!")
		break
	
	if length % p == 0:
		rlen = length / p
	else:
		rlen = length / p + 1

	cdef np.ndarray lrets = np.empty((rlen, nm), dtype=np.float64)
	cdef int i
	cdef int j
	cdef int iterr = 0
	for j in np.arange(nm):
		for i in np.arange(rlen):
			iterr = i + 1
			lrets[rlen - iterr, j] = values[length - (p * (iterr - 1) + 1), j]

	return lrets


cpdef np.ndarray rindex(np.ndarray index, int p):
	cdef int length = index.shape[0]
	cdef int rlen

	if length < 0:
		raise ValueError("The length of Array is shorter than requested period!")
		break

	if length % p == 0:
		rlen = length / p
	else:
		rlen = length / p + 1

	cdef np.ndarray pindex = np.empty(rlen, dtype=index.dtype)

	cdef int i
	cdef int iterr = 0
	for i in np.arange(rlen):
		iterr = i + 1
		pindex[rlen - iterr] = index[length - (p * (iterr - 1) + 1)]

	return pindex

# revise pframe with static periods
def parray(pframe, p=1):
	pdata = pframe.values
	cols = pframe.columns
	idx = pframe.index.values

	pdata = rarray(pdata, p)
	idx = rindex(idx, p)

	rpframe = df(data=pdata, columns=cols, index=idx)

	return rpframe 

def lrFrame(pframe):
	rdata = np.log(pframe / pframe.shift(1))
	return rdata





def mkTSband(tsignal, window=250, amultiple=1.5):
	""" 
		window: rolling window
		amultiple: Unit of sigma
	"""


	sma = tsignal.rolling(window=window).mean()

	sig = tsignal.rolling(window=window).std()

	tplus = sma + amultiple*sig

	tminus = sma - amultiple*sig

	sma = sma.dropna()
	tplus = tplus.dropna()
	tminus = tminus.dropna()

	return (sma, tplus, tminus)


def mkzTrend(tsignal, window=250):
	""" 
		tsignal: tracking signal
		window: rolling window
	"""

	zscore = (tsignal - tsignal.rolling(window=window).mean()) / tsignal.rolling(window=window).std()

	zscore = zscore.dropna()
	return zscore


def tsSig(tsignal, returns, lP=120):
	"""
		tsignal: invest signal
		returns: 
		lP	
	"""

	signal = tsignal.shift(1)
	signal = signal.dropna()

	rets = returns.ix[signal.index[0]:]

	signal = signal.astype(np.int32)

	rsig = rets * signal

	sign = rsig > 0
	sign = sign.astype(np.int32)

	sig = sign.cumsum() / signal.cumsum()
	
	sig = sig.ix[lP:]
	return sig

def mkquintile(factor, returns, ascend=False):
	"""
		크기가 같은 팩터 시계열, 수익률 시계열을 받아서 5분위 동일가중 시계열 도출
		수익률 시계열에 팩터 시계열을 맞춰야함(1시점씩 늦게)
		factor: factor 시계열(n X k)
		returns: 수익률 시계열(n X k)
		ascend: 오름차순(기본은 내림차순, False)
	"""
	if factor.shape != returns.shape:
		raise ValueError

	qrets = np.zeros((len(returns), 5))

	if not ascend: # 내림차순
		for i in np.arange(len(factor)):
			ftemp = factor.iloc[i]
			q = []
			for j in np.arange(5):
				q.append((4 - j) * .2)

			qm = []
			for j in np.arange(len(q)):
				if j == 0:
					qm.append(ftemp.loc[ftemp >= ftemp.quantile(q[j])].index.values)
				elif j <= (len(q) - 1):
					qm.append(ftemp.loc[(ftemp >= ftemp.quantile(q[j])) & (ftemp < ftemp.quantile(q[j - 1]))].index.values)
				elif j == (len(q) - 1):
					qm.append(ftemp.loc[(ftemp < ftemp.quantile(q[j - 1]))].index.values)

			for j in np.arange(len(qm)):
				qrets[i, j] = returns[qm[j]].iloc[i].mean()
	elif ascend: # 오름차순
		for i in np.arange(len(factor)):
			ftemp = factor.iloc[i]
			q = []
			for j in np.arange(5):
				q.append((j + 1) * .2)

			qm = []
			for j in np.arange(len(q)):
				if j == 0:
					qm.append(ftemp.loc[ftemp <= ftemp.quantile(q[j])].index.values)
				elif j <= (len(q) - 1):
					qm.append(ftemp.loc[(ftemp <= ftemp.quantile(q[j])) & (ftemp > ftemp.quantile(q[j - 1]))].index.values)
				elif j == (len(q) - 1):
					qm.append(ftemp.loc[ftemp >= ftemp.quantile(q[j - 1])].index.values)

			for j in np.arange(len(qm)):
				qrets[i, j] = returns[qm[j]].iloc[i].mean()

	qrets = df(qrets, index=returns.index, columns=['P1', 'P2', 'P3', 'P4', 'P5'])
	return qrets

cpdef np.ndarray rmonth(dates):
	"""
		dates: dataframe이나 Series의 날짜로 이뤄진 인덱스
	"""
	cdef int length
	cdef int i
	cdef np.ndarray f
	length = dates.shape[0]

	if length < 2:
		raise ValueError("The length of Date Array is shorter than requested period!")
		break
	f = np.array([])
	for i in np.arange(len(dates)):
		if i < len(dates) - 1:
			t = dates[i].to_pydatetime()
			tp = dates[i + 1].to_pydatetime()
			if t.month != tp.month:
				f = np.append(f, dates[i])
		else:
			f = np.append(f, dates[i])
	return f

# 분기로 뽑기
cpdef np.ndarray rquart(dates):
	"""
		dates: dataframe이나 Series의 날짜로 이뤄진 인덱스
	"""
	cdef np.ndarray mdates
	cdef int i
	cdef np.ndarray f

	mdates = rmonth(dates)

	f = np.array([])
	for i in np.arange(len(mdates)):
		temp = mdates[i].to_pydatetime()
		if temp.month == 3 or temp.month == 6 or temp.month == 9 or temp.month == 12:
			f = np.append(f, temp)

	return f


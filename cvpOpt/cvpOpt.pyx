# cvxopt를 사용한 포트폴리오 최적화 코드

import numpy as np
import pandas as pd
cimport numpy as np
from pandas import DataFrame as df
import cvxopt as opt
from cvxopt import solvers, blas

cdef class rpOpt:
	
	
	cdef int __n # 포트폴리오 구성종목 개수
	cdef np.ndarray __cvmat # 포트폴리오 공분산 행렬

	def __init__(self, cvmat, maxiters = 1000, abstol = 1e-6, reltol = 1e-6, feastol = 1e-6):
		"""
			cvmat: covarinace matrix(numpy.ndarray)
			maxiters: max iteration #
			etc.
		"""
		self.__cvmat = cvmat
		self.__n = self.__cvmat.shape[0]

		solvers.options['maxiters'] = maxiters
		solvers.options['abstol'] = abstol
		solvers.options['reltol'] = reltol
		solvers.options['feastol'] = feastol
		solvers.options['refinement'] = 5




	def __del__(self):
		pass


	def optimize(self):	
		def F(x=None, z=None):
			covmat = opt.matrix(self.__cvmat)
			if x is None: return 0, opt.matrix(1.0/self.__n, (self.__n, 1))
			if min(x) <= 0.0: return None
			f = x.T * covmat * x - sum(np.log(x))
			Df = (2 * covmat * x - (x ** -1)).T
			len(x)
			if z is None: return f, Df
			H = opt.spdiag(z[0]*(2 * covmat * opt.matrix(1.0, (self.__n, 1)) + x**-2))
			return f, Df, H
		A = opt.matrix(1.0, (1, self.__n))
		b = opt.matrix(1.0)
#		return solvers.cp(F, A=A, b=b)
		return solvers.cp(F)

	cpdef set_cvmat(self, cvmat):
		""" 공분산 행렬 설정 """
		self.__cvmat = cvmat
		self.__n = self.__cvmat.shape[0]

	cpdef get_cvmat(self):
		""" 입력된 공분산 행렬 반환 """ 
		return self.__cvmat

	def solution(self):
		sol = self.optimize()
		x = sol['x']
		x = np.array(x)
		x /= x.sum()
		return x
	
	cpdef double optVar(self):
		sol = self.optimize()
		x = np.array(sol['x'])
		x /= x.sum()
		cdef double vmin = 0
		vmin = x.transpose().dot(self.__cvmat).dot(x)
		return vmin

	# 각 투자비중에 대한 RRC 확인
	cpdef np.ndarray inRRC(self):
		sol = self.optimize()
		x = np.array(sol['x'])
		x /= x.sum()
		rcVec = x.transpose().dot(self.__cvmat) *x.transpose() / np.sqrt(self.optVar())
		rrcVec = rcVec / np.sqrt(self.optVar())
		return rrcVec

cdef class mvpOpt:
	
	solvers.options['maxiters'] = 10000
	solvers.options['abstol'] = 1e-10
	solvers.options['reltol'] = 1e-10
	solvers.options['feastol'] = 1e-10
	solvers.options['refinement'] = 5

	cdef int __n # 포트폴리오 구성종목 개수
	cdef np.ndarray __cvmat # 포트폴리오 공분산 행렬
	cdef np.ndarray __mVec
	cdef double __mMu

	# cvmat: covariance matrix, muVec: vector of expected Return
	def __init__(self, cvmat, muVec, minMu, maxiters = 1000, abstol = 1e-6, reltol = 1e-6, feastol = 1e-6):
		"""
			cvmat: Covariancde Matrix(np.ndarray)
			muVec: A vector of expecte return
			maxiter: maximum # of iteration
			minMu: Minimum requested expected return
		"""
			
		self.__cvmat = cvmat
		self.__n = self.__cvmat.shape[0]
		self.__mVec = muVec
		self.__mMu = minMu

		solvers.options['maxiters'] = maxiters
		solvers.options['abstol'] = abstol
		solvers.options['reltol'] = reltol
		solvers.options['feastol'] = feastol
		solvers.options['refinement'] = 5




	def __del__(self):
		pass

	def optimize(self):	
		covmat = opt.matrix(self.__cvmat)
		P = 2 * covmat
		q = opt.matrix(0.0, (self.__n, 1))
		
		# 제약조건 계수행렬
		gs = np.zeros((self.__n + 1, self.__n))
		gs[0, :] = self.__mVec
		gs[1:, :] = np.eye(self.__n)
		G = -opt.matrix(gs)
		
		# 제약값
		hs = np.zeros((self.__n + 1, 1))
		hs[0] = -self.__mMu
		h = opt.matrix(hs)
		A = opt.matrix(1.0, (1, self.__n))
		b = opt.matrix(1.0)
#		return solvers.cp(F, A=A, b=b)
		return solvers.qp(P, q, G=G, h=h, A=A, b=b)

	cpdef set_cvmat(self, cvmat):
		""" 공분산 행렬 설정 """
		self.__cvmat = cvmat
		self.__n = self.__cvmat.shape[0]

	cpdef get_cvmat(self):
		""" 입력된 공분산 행렬 반환 """ 
		return self.__cvmat

	cpdef set_minMu(self, minMu):
		""" 최소요구수익률 설정 """
		self.__mMu = minMu

	cpdef set_mVec(self, muVec):
		""" 기대수익률 벡터 설정 """
		self.__mVec = muVec

	cpdef get_minMu(self):
		""" 최소요구수익률 반환 """
		return self.__mMu

	cpdef get_mVec(self):
		""" 기대수익률 벡터 반환 """
		return self.__mVec

	def solution(self):
		sol = self.optimize()
		x = sol['x']
		x = np.array(x)
	
		return x
	
	cpdef double optVar(self):
		sol = self.optimize()
		x = np.array(sol['x'])
		
		cdef double vmin = 0
		vmin = x.transpose().dot(self.__cvmat).dot(x)
		return vmin

	# 각 투자비중에 대한 RRC 확인
	cpdef np.ndarray inRRC(self):
		sol = self.optimize()
		x = np.array(sol['x'])
		
		rcVec = x.transpose().dot(self.__cvmat) *x.transpose() / np.sqrt(self.optVar())
		rrcVec = rcVec / np.sqrt(self.optVar())
		return rrcVec

cdef class srmpOpt:
	
	solvers.options['maxiters'] = 10000
	solvers.options['abstol'] = 1e-10
	solvers.options['reltol'] = 1e-10
	solvers.options['feastol'] = 1e-10
	solvers.options['refinement'] = 5

	cdef int __n # 포트폴리오 구성종목 개수
	cdef np.ndarray __cvmat # 포트폴리오 공분산 행렬
	cdef np.ndarray __mVec

	# cvmat: covariance matrix, muVec: vector of expected Return
	def __init__(self, cvmat, muVec, maxiters = 10000, abstol = 1e-6, reltol = 1e-6, feastol = 1e-6):
		"""
			A class of Sharpe Ratio Maximization Portfolio
			Input data like cvmat, muVec should be based on 
			excess return against BM(riskfree or index)

			cvmat: Covariancde Matrix(np.ndarray)
			muVec: A vector of expecte return
			maxiter: maximum # of iteration(min: 10000)
		"""
			
		self.__cvmat = cvmat
		self.__n = self.__cvmat.shape[0]
		self.__mVec = muVec

		solvers.options['maxiters'] = maxiters
		solvers.options['abstol'] = abstol
		solvers.options['reltol'] = reltol
		solvers.options['feastol'] = feastol
		solvers.options['refinement'] = 5

	def __del__(self):
		pass

	def optimize(self):	
		covmat = opt.matrix(self.__cvmat)
		P = 2 * covmat
		q = opt.matrix(0.0, (self.__n, 1)) # 1차식의 계수값 벡터
		
		# 제약조건 계수행렬
		gs = np.zeros((self.__n, self.__n))
		gs[:, :] = np.eye(self.__n)
		G = -opt.matrix(gs)
		
		# 제약값
		hs = np.zeros((self.__n, 1))
		h = opt.matrix(hs)
		A = opt.matrix(self.__mVec, (1, self.__n))
		b = opt.matrix(1.0)
#		return solvers.cp(F, A=A, b=b)
		return solvers.qp(P, q, G=G, h=h, A=A, b=b)

	cpdef set_cvmat(self, cvmat):
		""" 공분산 행렬 설정 """
		self.__cvmat = cvmat
		self.__n = self.__cvmat.shape[0]

	cpdef get_cvmat(self):
		""" 입력된 공분산 행렬 반환 """ 
		return self.__cvmat

	cpdef set_minMu(self, minMu):
		""" 최소요구수익률 설정 """
		self.__mMu = minMu

	cpdef set_mVec(self, muVec):
		""" 기대수익률 벡터 설정 """
		self.__mVec = muVec

	cpdef get_minMu(self):
		""" 최소요구수익률 반환 """
		return self.__mMu

	cpdef get_mVec(self):
		""" 기대수익률 벡터 반환 """
		return self.__mVec

	def solution(self):
		sol = self.optimize()
		x = sol['x']
		x = np.array(x)

		x = x/x.sum()
	
		return x
	
	cpdef double optVar(self):
		sol = self.optimize()
		x = np.array(sol['x'])
		
		cdef double vmin = 0
		vmin = x.transpose().dot(self.__cvmat).dot(x)
		return vmin

	# 각 투자비중에 대한 RRC 확인
	cpdef np.ndarray inRRC(self):
		sol = self.optimize()
		x = np.array(sol['x'])
		
		rcVec = x.transpose().dot(self.__cvmat) *x.transpose() / np.sqrt(self.optVar())
		rrcVec = rcVec / np.sqrt(self.optVar())
		return rrcVec


	

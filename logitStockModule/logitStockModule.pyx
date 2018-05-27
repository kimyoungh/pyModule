# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:04:38 2017

@author: yh.kim
가치 모형 모듈
"""
import numpy as np
cimport numpy as np
from pandas import DataFrame as df
from datetime import datetime as dt
import pandas as pd
from arch import arch_model
import xlwings as xw
import quantM as qnt
import pytsignal as pt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

cpdef simpleGarch(returns, p = 250, r = 20, rday='month'):
    """
        단일 주가(지수) 정보만 이용한 GARCH 모델을 만들고
        제공받은 수익률 데이터의 가용한 모든 기간을 활용해서 롤링 윈도우 방식으로
        기대변동성, 시기별 계수 구해서 반환
        returns: 단일 주가(지수)의 로그수익률 데이터프레임
        p: 회귀 기간(기본: 250일)
        r: 변동성 계산 기간
        rday: 산출 시점 단위(기본: 월(month)), 일로 하고 싶으면 'day' 입력
    """
    cdef int i

    returns = df(returns)
    sigma = returns.rolling(r).std(ddof=1).dropna()
    fsigma = sigma.shift(-1).dropna() # 다음날 변동성

    lagreturns = returns.shift(1).dropna()
    returns = returns.loc[lagreturns.index] 
    cdef np.ndarray sigsq
    cdef np.ndarray coef
    
    if rday == 'month':
        rdays = pt.rmonth(sigma.iloc[p-1:].index)
    elif rday == 'day':
        rdays = sigma.index

    sigsq = np.array([])
    coef = np.zeros((len(rdays)-1, 2))
    
    for i in np.arange(len(rdays)-1):
        t = np.argwhere(returns.index == rdays[i])[0][0]
        st = np.argwhere(sigma.index == rdays[i])[0][0]
        lm = linear_model.LinearRegression()
    
        lagrtemp = df(lagreturns.iloc[t-p+1:t+1])
        rtemp = returns.iloc[t-p+1:t+1]
        
        armodel = lm.fit(lagrtemp.iloc[:-1], rtemp.iloc[:-1])
        eps = rtemp - armodel.predict(lagrtemp)
        
        X = df(eps.values, index=eps.index, columns=['epsilon'])
        X['sigma'] = sigma.loc[eps.index]
        X = X ** 2
        fstemp = fsigma ** 2
        glm = linear_model.LinearRegression()
        garch = glm.fit(X.iloc[:-1], fstemp.loc[X.iloc[:-1].index])
        
        sigsq = np.append(sigsq, garch.predict(X.iloc[-1].values.reshape(1, 2)))
        coef[i] = garch.coef_
    rsigsq = df()
    rcoef = df()
    rsigsq = df(sigsq, columns=['Ex(Sigma)'], index=rdays[:-1])
    sig = np.sqrt(rsigsq)
    rsig = sig.loc[rdays].shift(1)
    rsig['Sigma(Actual)'] = sigma.loc[rsig.index]
    rcoef = df(coef, index=rsigsq.index, columns=['eps Coef', 'sig Coef'])
    
    return (sig, rsig, rcoef)

cpdef influGarch(returns, lag=np.array([1, 0]), p=250, r=20, rday='month'):
    """
        타겟 국가 뿐만 아니라 영향을 주는 다른 지수들을 이용해서 변형 GARCH 모형 만들기
        제공받은 수익률 데이터의 가용한 모든 기간을 활용해서 롤링 윈도우 방식으로
        기대변동성, 시기별 계수 구해서 반환
        returns: 주가(지수) 수익률 데이터프레임(반드시 첫번째 종목이 타겟, 나머지가 영향을 끼치는 애들)
        lag: np.ndarray인데, 첫번째 값부터가 returns의 컬럼순으로 래깅 몇일 하는지 여부
        p: 회귀 기간(기본: 250일)
        r: 변동성 계산 기간
        rday: 산출 시점 단위(기본: 월(month)), 일로 하고 싶으면 'day' 입력
    """  
    
    cdef int i
    cdef int j
    sigma = returns.rolling(r).std(ddof=1).dropna()
    fsigma = sigma.shift(-1).dropna() # 다음날 변동성
    
    nations = returns.columns.values
    lagreturns = {}
    lagreturns[nations[0]] = returns[nations[0]].shift(1).dropna() # 첫번째껀 무조건 1일 래깅
    for j in np.arange(len(lag)): 
        lagreturns[nations[j + 1]] = returns[nations[j + 1]].shift(lag[j] + 1).dropna()
        
    cdef np.ndarray sigsq
    cdef np.ndarray coef
    
    if rday == 'month':
        rdays = pt.rmonth(sigma.iloc[p-1:].index)
    elif rday == 'day':
        rdays = sigma.index
    
    sigsq = np.array([])
    coef = np.zeros((len(rdays)-1, len(lag)+2))
    
    returns = returns.loc[lagreturns[nations[0]].index]
    
    cdef dict lms
    cdef dict armodel
    cdef dict lagrtemp
    cdef dict rtemp
    cdef dict eps
    
    for i in np.arange(len(rdays) - 1): 
        t = np.argwhere(returns.index == rdays[i])[0][0]
        st = np.argwhere(sigma.index == rdays[i])[0][0]
        lms = {}
        armodel = {}
        lagrtemp = {}
        rtemp = {}
        eps = {}
        
        lms[nations[0]] = linear_model.LinearRegression()
        lagrtemp[nations[0]] = df(lagreturns[nations[0]].iloc[t-p+1:t+1])
        rtemp[nations[0]] = df(returns[nations[0]].iloc[t-p+1:t+1])
        for j in np.arange(1, len(returns.columns)):
            lms[nations[j]] = linear_model.LinearRegression()    
            lagrtemp[nations[j]] = df(lagreturns[nations[j]].iloc[t-p+1:t+1])
            if lag[j-1] == 0:
                rtemp[nations[j]] = df(returns[nations[j]].iloc[t-p+1:t+1])
            else:
                rtemp[nations[j]] = df(returns[nations[j]].iloc[t-p -lag[j-1]+1:t+1].shift(1).dropna())
            
        X = df()
        
        for j in np.arange(len(returns.columns)):
            armodel[nations[j]] = lms[nations[j]].fit(lagrtemp[nations[j]].iloc[:-1], rtemp[nations[j]].iloc[:-1])
            eps[nations[j]] = rtemp[nations[j]] - armodel[nations[j]].predict(lagrtemp[nations[j]])
        
        X = df(eps[nations[0]].values, columns=[nations[0]], index=eps[nations[0]].index.values)
        for j in np.arange(len(lag)):
            X[nations[j + 1]] = eps[nations[j+1]].values
        X['sigma'] = sigma[nations[0]].loc[eps[nations[0]].index]
        
        X = X ** 2
        fstemp = fsigma[nations[0]] ** 2
        
        glm = linear_model.LinearRegression()
        garch = glm.fit(X.iloc[:-1], fstemp.loc[X.iloc[:-1].index])
        sigsq = np.append(sigsq, garch.predict(X.iloc[-1].values.reshape(1, len(lag)+2)))
        coef[i] = garch.coef_
    rsigsq = df()
    rcoef = df()
    rsigsq = df(sigsq, columns=['Ex(Sigma)'], index=rdays[:-1])
    sig = np.sqrt(rsigsq)
    rsig = sig.loc[rdays].shift(1)
    rsig['Sigma(Actual)'] = sigma[nations[0]].loc[rsig.index]
    rcoef = df(coef, index=rsigsq.index, columns=X.columns)
    
    return (sig, rsig, rcoef)


class interGarch:
    """
        단순 GARCH 및 다른 종목, 국가의 잔차까지 고려하는 GARCH 모형 클래스
        price: 분석 대상 국가(종목)의 지수(주가) 데이터프레임
        otherprice: 분석 대상 국가(종목)에 영향을 미치는 국가(종목)의 지수(주가) 데이터프레임
        p: 회귀 기간(기본: 250일)
        r = 변동성 계산 기간
        rday: 산출 시점 단위(기본: 월(month)), 일로 하고 싶으면 'day' 입력
    """

    __rets = df() # 분석 대상 국가(종목)의 로그수익률 데이터프레임
    __otherrets = df() # 분석 대상 국가(종목)에 영향을 미치는 국가들의 로그수익률 데이터프레임
    __p = 250
    __r = 20
    __rday = 'month'
    
    def __init__(self, price, otherprice=df(), p=250, r=20, rday='month'):
        rtemp = np.log(price / price.shift(1)).dropna()
        otemp = np.log(otherprice / otherprice.shift(1)).dropna()
        self.__rets = rtemp
        self.__otherrets = otemp
        self.__p = p
        self.__r = r
        self.__rday = rday
    
    def simpleGarch(self):
        """
            분석 대상 국가(종목)만 고려한 GARCH 모델 결과 도출
        """
        return simpleGarch(self.__rets, p=self.__p, r=self.__r, rday=self.__rday)
    
    def influGarch(self, lag = np.array([1, 0])):
        """
            분석 대상 국가(종목) 및 다른 국가들 수익률 고려한 GARCH 모델 결과 도출
            다른 국가들은 동일하게 수익률 고려해서 결과 도출
        """
        k = self.__rets.copy()
        k[self.__otherrets.columns] = self.__otherrets.copy()
        
        return influGarch(k, lag = lag, p=self.__p, r=self.__r, rday=self.__rday)
    
    def ordinalGarch(self, lag = np.array([1, 0])):
        """
            분석 대상 국가(종목)에 영향을 미치는 국가(종목) 간의 관계성을 반영한 GARCH
            분석대상 국가(종목) 이외의 국가(종목)은 타 국가(종목)에 영향 받지 않는 순으로 
            column으로 나열(예: 한국, 미국, 일본...순서로)
            lag은 lag
        """
        
        returns = self.__rets.copy()
        returns[self.__otherrets.columns] = self.__otherrets.copy()
        
        sigma = returns.rolling(self.__r).std(ddof=1).dropna()
        fsigma = sigma.shift(-1).dropna() # 다음날 변동성
        
        nations = returns.columns.values
        lagreturns = {}
        lagreturns[nations[0]] = returns[nations[0]].shift(1).dropna() # 첫번째껀 무조건 1일 래깅
        for j in np.arange(len(lag)):
            lagreturns[nations[j + 1]] = df(returns[nations[j + 1]].shift(lag[j] + 1))

        if self.__rday == 'month':
            rdays = pt.rmonth(sigma.iloc[self.__p-1:].index)
        elif self.__rday == 'day':
            rdays = sigma.index
            
        sigsq = np.array([])
        coef = np.zeros((len(rdays) - 1, len(lag) + 2))
        
        returns  = returns.loc[lagreturns[nations[0]].index]
               
        lms = {}
        armodel = {}
        lagrtemp = {}
        rtemp = {}
        eps = {}
        
        for i in np.arange(len(rdays) - 1):
            t = np.argwhere(returns.index == rdays[i])[0][0]
            st = np.argwhere(sigma.index == rdays[i])[0][0]
            
            lms[nations[0]] = linear_model.LinearRegression()
            lagrtemp[nations[0]] = df(lagreturns[nations[0]].iloc[t-self.__p+1:t+1])
            rtemp[nations[0]] = df(returns[nations[0]].iloc[t-self.__p:t+1])
            for j in np.arange(1, len(returns.columns)):
                lms[nations[j]] = linear_model.LinearRegression()
                lagrtemp[nations[j]] = df(lagreturns[nations[j]].iloc[t-self.__p+1:t+1])
                if lag[j-1] == 0:
                    rtemp[nations[j]] = df(returns[nations[j]].iloc[t-self.__p+1:t+1])
                else:
                    rtemp[nations[j]] = df(returns[nations[j]].iloc[t-self.__p-lag[j-1]+1:t+1].shift(1).dropna())
        
            X = df()
        
            # armodel 만들기
            armodel[nations[0]] = lms[nations[0]].fit(lagrtemp[nations[0]].iloc[:-1], rtemp[nations[0]].iloc[:-1])
            eps[nations[0]] = rtemp[nations[0]] - armodel[nations[0]].predict(lagrtemp[nations[0]])
            
            lagchain = df()
            for j in np.arange(len(returns.columns)-1):
                if j == 0:
                    armodel[nations[j + 1]] = lms[nations[j + 1]].fit(lagrtemp[nations[j + 1]].iloc[:-1], 
                           rtemp[nations[j + 1]].iloc[:-1])
                    eps[nations[j + 1]] = rtemp[nations[j + 1]] - armodel[nations[j + 1]].predict(lagrtemp[nations[j + 1]])
                else:
                    lagchain = df(lagrtemp[nations[j + 1]])
                    for k in np.arange(1, (j + 1)):
                        lagchain[nations[k]] = lagrtemp[nations[k]]

                    armodel[nations[j + 1]] = lms[nations[j + 1]].fit(lagchain.iloc[:-1],
                           rtemp[nations[j + 1]].iloc[:-1])
                    eps[nations[j + 1]] = rtemp[nations[j + 1]] - armodel[nations[j + 1]].predict(lagchain)
                    
            X = df(eps[nations[0]].values, columns=[nations[0]], index=eps[nations[0]].index.values)
            for j in np.arange(len(lag)):
                X[nations[j + 1]] = eps[nations[j + 1]].values
            X['sigma'] = sigma[nations[0]].loc[eps[nations[0]].index]
            
            X = X ** 2
            fstemp = fsigma[nations[0]] ** 2
            
            glm = linear_model.LinearRegression()
            garch = glm.fit(X.iloc[:-1], fstemp.loc[X.iloc[:-1].index])
            sigsq = np.append(sigsq, garch.predict(X.iloc[-1].values.reshape(1, len(lag) + 2)))
            coef[i] = garch.coef_
        
        rsigsq = df()
        rcoef = df()
        rsigsq = df(sigsq, columns=['Ex(Sigma)'], index=rdays[:-1])
        sig = np.sqrt(rsigsq)
        rsig = sig.loc[rdays].shift(1)
        rsig['Sigma(Actual)'] = sigma[nations[0]].loc[rsig.index]
        rcoef = df(coef, index=rsigsq.index, columns=X.columns)
        
        return (sig, rsig, rcoef)
        
    def getReturns(self):
        return self.__rets
    
    def getotherReturns(self):
        return self.__otherrets
    

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:58:35 2017

@author: yh.kim
Spyder나 GUI IDE에서 작성 편집하기 좋음
"""
import numpy as np
cimport numpy as np
import pandas as pd
from pandas import DataFrame as df
from datetime import datetime as dt
from sklearn.decomposition import PCA
import pytsignal as pt
import scipy.stats as ss

def mkquantile(factor, returns, ascend=False, p=5):
    """
    크기가 같은 팩터 시계열, 수익률 시계열을 받아서
    p분위 동일가중 시계열 도출
    수익률 시계열에 팩터 시계열을 맞춰야함(1시점씩 늦게)
    factor: factor 시계열(n X k) 데이터프레임
    returns: 수익률 시계열(n X k) 데이터프레임
    ascend: 오름차순(기본은 내림차순, False)
    p: 몇분위로 나눌건지.. 기본은 5분위
    """
    if factor.shape != returns.shape:
        raise ValueError("factor와 returns의 배열 크기가 달라요!")
        
    q = cmkquantile(factor.values, returns.values, ascend=ascend, p = p)
    
    P = []
    for i in np.arange(p):
        P.append("P" + str(i + 1))
    
    qrets = df(q, index=returns.index, columns=P)

    return qrets

def pmkquantile(factor, returns, ascend=False, p=5, t = 0.05):
    """
    Precise mkQuantile
    
    극값을 제거한(양쪽 t만큼(ex 양쪽 5%씩 제거하고 싶으면 t=0.05)) 팩터 수익률 계산
    
    크기가 같은 팩터 시계열, 수익률 시계열을 받아서
    p분위 동일가중 시계열 도출
    수익률 시계열에 팩터 시계열을 맞춰야함(1시점씩 늦게)
    factor: factor 시계열(n X k) 데이터프레임
    returns: 수익률 시계열(n X k) 데이터프레임
    ascend: 오름차순(기본은 내림차순, False)
    p: 몇분위로 나눌건지.. 기본은 5분위
    """
    if factor.shape != returns.shape:
        raise ValueError("factor와 returns의 배열 크기가 달라요!")
        
    q = cpmkquantile(factor.values, returns.values, ascend=ascend, p = p, t = t)
    
    P = []
    for i in np.arange(p):
        P.append("P" + str(i + 1))
    
    qrets = df(q, index=returns.index, columns=P)

    return qrets


cpdef np.ndarray cpmkquantile(np.ndarray factor, np.ndarray returns, ascend=False, int p = 5, double t = 0.05):
    """
    Precise mkQuantile
    
    극값을 제거한(양쪽 t만큼(ex 양쪽 5%씩 제거하고 싶으면 t=0.05)) 팩터 수익률 계산
    
    크기가 같은 팩터 시계열, 수익률 시계열을 받아서
    p분위 동일가중 시계열 도출
    수익률 시계열에 팩터 시계열을 맞춰야함(1시점씩 늦게)
    factor: factor 시계열(n X k) 데이터프레임
    returns: 수익률 시계열(n X k) 데이터프레임
    ascend: 오름차순(기본은 내림차순, False)
    p: 몇분위로 나눌건지.. 기본은 5분위
    """

    cdef np.ndarray qrets
    qrets = np.zeros((len(returns), p))
    
    cdef int i
    cdef int j
    cdef np.ndarray ftemp
    cdef list qm
    cdef list q
    cdef np.ndarray c
    cdef np.ndarray trims
    
    if not ascend: # 내림차순
        for i in np.arange(len(factor)):
            ftemp = factor[i]
            rtemp = returns[i]
            for j in np.arange(len(ftemp)):
                trims = np.argwhere((ftemp > np.percentile(ftemp, t * 100)) & 
                                    (ftemp < np.percentile(ftemp, (1 - t)*100)))[:, 0]
            ftemp = ftemp[trims]
            rtemp = rtemp[trims]
            q = []
            for j in np.arange(p):
                q.append((p - 1 - j) * 1./np.float(p))
                
            qm = []
            for j in np.arange(len(q)):
                if j == 0:
                    c = np.argwhere(ftemp >= np.percentile(ftemp, q[j] * 100))[:, 0]
                    qm.append(c)
                elif j <= (len(q) - 1):
                    c = np.argwhere((ftemp >= np.percentile(ftemp, q[j] * 100)) & (ftemp < np.percentile(ftemp, 
                                    q[j - 1] * 100)))[:, 0]
                    qm.append(c)
                elif j == (len(q) - 1):
                    c = np.argwhere(ftemp < np.percentile(ftemp, q[j - 1] * 100))[:, 0]
                    qm.append(c)
                    
            for j in np.arange(len(qm)):
                qrets[i, j] = rtemp[qm[j]].mean()
    elif ascend: # 오름차순
        for i in np.arange(len(factor)):
            ftemp = factor[i]
            rtemp = returns[i]
            for j in np.arange(len(ftemp)):
                trims = np.argwhere((ftemp > np.percentile(ftemp, t * 100)) & 
                                    (ftemp < np.percentile(ftemp, (1 - t)*100)))[:, 0]
            ftemp = ftemp[trims]
            rtemp = rtemp[trims]
            
            q = []
            for j in np.arange(p):
                q.append((j + 1) * 1./np.float(p))
            qm = []
            for j in np.arange(len(q)):
                if j == 0:
                    c = np.argwhere(ftemp <= np.percentile(ftemp, q[j] * 100))[:, 0]
                    qm.append(c)
                elif j <= (len(q) - 1):
                    c = np.argwhere((ftemp <= np.percentile(ftemp, q[j] * 100)) & (ftemp > np.percentile(ftemp, 
                                    q[j - 1] * 100)))[:, 0]
                    qm.append(c)
                elif j == (len(q) - 1):
                    c = np.argwhere(ftemp > np.percentile(ftemp, q[j - 1] * 100))[:, 0]
                
            for j in np.arange(len(qm)):
                qrets[i, j] = rtemp[qm[j]].mean()
    
    return qrets


cpdef np.ndarray cmkquantile(np.ndarray factor, np.ndarray returns, ascend=False, int p=5):
    """
    크기가 같은 팩터 시계열, 수익률 시계열을 받아서
    p분위 동일가중 시계열 도출
    수익률 시계열에 팩터 시계열을 맞춰야함(1시점씩 늦게)
    factor: factor 시계열(n X k)
    returns: 수익률 시계열(n X k)
    ascend: 오름차순(기본은 내림차순, False)
    p: 몇분위로 나눌건지.. 기본은 5분위
    """
    
    cdef np.ndarray qrets
    qrets = np.zeros((len(returns), p))
    
    cdef int i
    cdef int j
    cdef np.ndarray ftemp
    cdef list qm
    cdef list q
    cdef np.ndarray c
    
    if not ascend: # 내림차순
        for i in np.arange(len(factor)):
            ftemp = factor[i]
            q = []
            for j in np.arange(p):
                q.append((p - 1 - j) * 1./np.float(p))
                
            qm = []
            for j in np.arange(len(q)):
                if j == 0:
                    c = np.argwhere(ftemp >= np.percentile(ftemp, q[j] * 100))[:, 0]
                    qm.append(c)
                elif j <= (len(q) - 1):
                    c = np.argwhere((ftemp >= np.percentile(ftemp, q[j] * 100)) & (ftemp < np.percentile(ftemp, 
                                    q[j - 1] * 100)))[:, 0]
                    qm.append(c)
                elif j == (len(q) - 1):
                    c = np.argwhere(ftemp < np.percentile(ftemp, q[j - 1] * 100))[:, 0]
                    qm.append(c)
                    
            for j in np.arange(len(qm)):
                qrets[i, j] = returns[i, qm[j]].mean()
    elif ascend: # 오름차순
        for i in np.arange(len(factor)):
            ftemp = factor[i]
            q = []
            for j in np.arange(p):
                q.append((j + 1) * 1./np.float(p))
            qm = []
            for j in np.arange(len(q)):
                if j == 0:
                    c = np.argwhere(ftemp <= np.percentile(ftemp, q[j] * 100))[:, 0]
                    qm.append(c)
                elif j <= (len(q) - 1):
                    c = np.argwhere((ftemp <= np.percentile(ftemp, q[j] * 100)) & (ftemp > np.percentile(ftemp, 
                                    q[j - 1] * 100)))[:, 0]
                    qm.append(c)
                elif j == (len(q) - 1):
                    c = np.argwhere(ftemp > np.percentile(ftemp, q[j - 1] * 100))[:, 0]
                
            for j in np.arange(len(qm)):
                qrets[i, j] = returns[i, qm[j]].mean()
    
    return qrets

cpdef double abRatio(data, c=3):
    """
        data: 주성분분석을 실시할 시계열 np.ndarray
        c: 사용할 주성분 개수(디폴트: 3개)
        return ar(Absorption Ratio)
    """
    
    if data.shape[1] >= c:
        try:
            pca = PCA(n_components = data.shape[1])
            pca.fit(data)
            ar = pca.explained_variance_[:c].sum()/pca.explained_variance_.sum()
        except Exception as e:
            ar = np.nan
    else:
        ar = 1.
    
    return ar

def arChg(data, c=3, p=24, y=12):
    """
        AR의 변화율 산출
        최근 AR을 과거 y기간에 대해서 zscore 계산
        data: 주성분분석을 실시할 시계열 np.ndarray
        c: 사용할 주성분 개수(디폴트: 3개)
        p: rolling window 크기(디폴트: 24개)
        y: 1년이 몇개 row로 이뤄져있는지(디폴트: 12개, 월단위로 보는거징)
    """
    
    aar = carChg(data.values, c, p)
    
    aar = df(aar, index=data.index[p - 1:], columns=["deltaAR"])
    
    dar = (aar - aar.rolling(y).mean())/(aar.rolling(y).std())
    dar = dar.dropna()
    
    return dar
    


cpdef np.ndarray carChg(np.ndarray data, int c=3, int p=24):
    """
        AR의 변화율 산출 과정
        최근 AR을 과거 y기간에 대해서 zscore 계산
        data: 주성분분석을 실시할 시계열 np.ndarray
        c: 사용할 주성분 개수(디폴트: 3개)
        p: rolling window 크기(디폴트: 24개)
    """
    
    cdef int i
    cdef np.ndarray kar
    cdef double tt
    ar = []
    
    for i in np.arange(p - 1, len(data)):
        tt = abRatio(data[i - p + 1: i + 1], c)
        ar.append(tt)
    
    kar = np.array(ar)
    
    return kar


cpdef clogrebalancing(returns, wt, double bfee=0.0, double sfee=0.0):
    """
        Backtest에서 수수료 및 리밸런싱을 적용한
        바스켓 변화과정 계산하여 배열 반환
	returns: 투자 대상 종목들의 수익률 데이터프레임
	wt: 리밸런싱 투자비중 데이터, 리밸런싱 시점에 대한 데이터만 있음 데이터프레임
	bfee: 살때 수수료
	sfee: 팔떄 수수료(세금도 포함)

        wt 시점이 returns보다 한 시점 빠름.
	예: 2011년 1월30일 종가 기준으로 투자비중 설정한 월간 수익률 기준 포트폴리오에서 returns에서 실제로 곱해지는 날짜는 2011년 2월 28일임
    """

    cdef int i
    cdef int j
    cdef int t
    cdef np.ndarray retv
    cdef np.ndarray wv
     
    if bfee >= 0.1:
        bfee /= 100.
    if sfee >= 0.1:
        sfee /= 100.

    # wt를 returns 크기에 맞추기
    rett = returns.index
    wtt = wt.index
    wi = rett.append(wtt).unique()
    wi = wi.sort_values()
    wts = np.zeros((len(wi), returns.shape[1]))
    wts = df(wts, index=wi, columns=returns.columns)
    wts.loc[wt.index] = wt

    weights = wts.shift(1).dropna()
    wtemp = wt.loc[wi]
    wtemp = wtemp.shift(1).dropna()
    retv = returns.values.copy() # np.ndarray로 값 가져오기(속도 고려)
    
    wv = weights.values.copy()

    # nan 제거
    for i in np.arange(len(retv)):
        for j in np.arange(retv.shape[1]):
            if retv[i, j] - retv[i, j] != 0:
                retv[i, j] = 0

    if wv.shape[1] != retv.shape[1]:
        
        raise ValueError("Input Variable Size not equal!")

    # 리밸런싱 시점 위치 정보 가져오기
    cdef np.ndarray k
    k = np.array([])

    for i in np.arange(len(weights)):
        if weights.index[i] in wtemp.index:
            k = np.append(k, i)

    for i in np.arange(len(wv)):
        if i == 0:
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > 0:
                    retv[i, j] = np.exp(retv[i, j]) * (1 - bfee) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
                else:
                    retv[i, j] = np.exp(retv[i, j]) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)

        elif i != 0 and (i not in k):
            wv[i] = wv[i - 1] * np.exp(retv[i - 1])
            wv[i] /= wv[i].sum()
            retv[i] = np.exp(retv[i]) - 1.
            retv[i] = np.log(retv[i] + 1.)

        elif i != 0 and (i in k):
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > wv[i - 1, j]:
                    retv[i, j] = np.exp(retv[i, j]) * (1 - bfee) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
                elif wv[i, j] < wv[i -1, j]:
                    retv[i, j] = np.exp(retv[i, j]) * (1 - sfee) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
                else:
                    retv[i, j] = np.exp(retv[i, j]) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)


    cdef np.ndarray wr # 수수료가 적용된 수익률과 투자비중을 곱한 최종 결과    
    wr = retv * wv
    wreturns = df(wr, index=weights.index, columns=weights.columns)
    
    return wreturns
    
cpdef crebalancing(returns, wt, double bfee=0.0, double sfee=0.0):
    """
        Backtest에서 수수료 및 리밸런싱을 적용한
        바스켓 변화과정 계산하여 배열 반환
	returns: 투자 대상 종목들의 수익률 데이터프레임
	wt: 리밸런싱 투자비중 데이터, 리밸런싱 시점에 대한 데이터만 있음 데이터프레임
	bfee: 살때 수수료
	sfee: 팔떄 수수료(세금도 포함)

        wt 시점이 returns보다 한 시점 빠름.
	예: 2011년 1월30일 종가 기준으로 투자비중 설정한 월간 수익률 기준 포트폴리오에서 returns에서 실제로 곱해지는 날짜는 2011년 2월 28일임
    """

    cdef int i
    cdef int j
    cdef int t
    cdef np.ndarray retv
    cdef np.ndarray wv
     
    if bfee >= 0.1:
        bfee /= 100.
    if sfee >= 0.1:
        sfee /= 100.

    # wt를 returns 크기에 맞추기
    rett = returns.index
    wtt = wt.index
    wi = rett.append(wtt).unique()
    wi = wi.sort_values()
    wts = np.zeros((len(wi), returns.shape[1]))
    wts = df(wts, index=wi, columns=returns.columns)
    wts.loc[wt.index] = wt

    weights = wts.shift(1).dropna()
    wtemp = wt.loc[wi]
    wtemp = wtemp.shift(1).dropna()
    retv = returns.values.copy() # np.ndarray로 값 가져오기(속도 고려)
    
    wv = weights.values.copy()

    # nan 제거
    for i in np.arange(len(retv)):
        for j in np.arange(retv.shape[1]):
            if retv[i, j] - retv[i, j] != 0:
                retv[i, j] = 0

    if wv.shape[1] != retv.shape[1]:
        
        raise ValueError("Input Variable Size not equal!")

    # 리밸런싱 시점 위치 정보 가져오기
    cdef np.ndarray k
    k = np.array([])

    for i in np.arange(len(weights)):
        if weights.index[i] in wtemp.index:
            k = np.append(k, i)

    for i in np.arange(len(wv)):
        if i == 0:
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > 0:
                    retv[i, j] = (1 + retv[i, j]) * (1 - bfee) - 1.
                else:
                    retv[i, j] = (1 + retv[i, j]) - 1.

        elif i != 0 and (i not in k):
            wv[i] = wv[i - 1] * (1 + retv[i - 1])
            wv[i] /= wv[i].sum()
            retv[i] = (1 + retv[i]) - 1.

        elif i != 0 and (i in k):
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > wv[i - 1, j]:
                    retv[i, j] = (1 + retv[i, j]) * (1 - bfee) - 1.
                elif wv[i, j] < wv[i -1, j]:
                    retv[i, j] = (1 + retv[i, j]) * (1 - sfee) - 1.
                else:
                    retv[i, j] = (1 + retv[i, j]) - 1.


    cdef np.ndarray wr # 수수료가 적용된 수익률과 투자비중을 곱한 최종 결과    
    wr = retv * wv
    wreturns = df(wr, index=weights.index, columns=weights.columns)
    
    return wreturns


cpdef chlogweights(returns, wt, double bfee=0.0, double sfee=0.0):
    """
        Backtest에서 수수료 및 리밸런싱을 적용한
        바스켓 변화과정 계산하여 투자비중 변화 시계열 반환
	returns: 투자 대상 종목들의 수익률 데이터프레임
	wt: 리밸런싱 투자비중 데이터, 리밸런싱 시점에 대한 데이터만 있음 데이터프레임
	bfee: 살때 수수료
	sfee: 팔떄 수수료(세금도 포함)

        wt 시점이 returns보다 한 시점 빠름.
	예: 2011년 1월30일 종가 기준으로 투자비중 설정한 월간 수익률 기준 포트폴리오에서 returns에서 실제로 곱해지는 날짜는 2011년 2월 28일임
    """

    cdef int i
    cdef int j
    cdef int t
    cdef np.ndarray retv
    cdef np.ndarray wv
     
    if bfee >= 0.1:
        bfee /= 100.
    if sfee >= 0.1:
        sfee /= 100.

    # wt를 returns 크기에 맞추기
    rett = returns.index
    wtt = wt.index
    wi = rett.append(wtt).unique()
    wi = wi.sort_values()
    wts = np.zeros((len(wi), returns.shape[1]))
    wts = df(wts, index=wi, columns=returns.columns)
    wts.loc[wt.index] = wt

    weights = wts.shift(1).dropna()
    wtemp = wt.loc[wi]
    wtemp = wtemp.shift(1).dropna()
    retv = returns.values.copy() # np.ndarray로 값 가져오기(속도 고려)
    
    wv = weights.values.copy()

    # nan 제거
    for i in np.arange(len(retv)):
        for j in np.arange(retv.shape[1]):
            if retv[i, j] - retv[i, j] != 0:
                retv[i, j] = 0

    if wv.shape[1] != retv.shape[1]:
        
        raise ValueError("Input Variable Size not equal!")

    # 리밸런싱 시점 위치 정보 가져오기
    cdef np.ndarray k
    k = np.array([])

    for i in np.arange(len(weights)):
        if weights.index[i] in wtemp.index:
            k = np.append(k, i)

    for i in np.arange(len(wv)):
        if i == 0:
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > 0:
                    retv[i, j] = np.exp(retv[i, j]) * (1 - bfee) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
                else:
                    retv[i, j] = np.exp(retv[i, j]) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)

        elif i != 0 and (i not in k):
            wv[i] = wv[i - 1] * np.exp(retv[i - 1])
            wv[i] /= wv[i].sum()
            retv[i] = np.exp(retv[i]) - 1.
            retv[i] = np.log(retv[i] + 1.)

        elif i != 0 and (i in k):
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > wv[i - 1, j]:
                    retv[i, j] = np.exp(retv[i, j]) * (1 - bfee) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
                elif wv[i, j] < wv[i -1, j]:
                    retv[i, j] = np.exp(retv[i, j]) * (1 - sfee) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
                else:
                    retv[i, j] = np.exp(retv[i, j]) - 1.
                    retv[i, j] = np.log(retv[i, j] + 1.)
    
    return wv

cpdef chweights(returns, wt, double bfee=0.0, double sfee=0.0):
    """
        Backtest에서 수수료 및 리밸런싱을 적용한
        바스켓 변화과정 계산하여 투자비중 변화 시계열 반환
	returns: 투자 대상 종목들의 수익률 데이터프레임
	wt: 리밸런싱 투자비중 데이터, 리밸런싱 시점에 대한 데이터만 있음 데이터프레임
	bfee: 살때 수수료
	sfee: 팔떄 수수료(세금도 포함)

        wt 시점이 returns보다 한 시점 빠름.
	예: 2011년 1월30일 종가 기준으로 투자비중 설정한 월간 수익률 기준 포트폴리오에서 returns에서 실제로 곱해지는 날짜는 2011년 2월 28일임
    """

    cdef int i
    cdef int j
    cdef int t
    cdef np.ndarray retv
    cdef np.ndarray wv
     
    if bfee >= 0.1:
        bfee /= 100.
    if sfee >= 0.1:
        sfee /= 100.

    # wt를 returns 크기에 맞추기
    rett = returns.index
    wtt = wt.index
    wi = rett.append(wtt).unique()
    wi = wi.sort_values()
    wts = np.zeros((len(wi), returns.shape[1]))
    wts = df(wts, index=wi, columns=returns.columns)
    wts.loc[wt.index] = wt

    weights = wts.shift(1).dropna()
    wtemp = wt.loc[wi]
    wtemp = wtemp.shift(1).dropna()
    retv = returns.values.copy() # np.ndarray로 값 가져오기(속도 고려)
    
    wv = weights.values.copy()

    # nan 제거
    for i in np.arange(len(retv)):
        for j in np.arange(retv.shape[1]):
            if retv[i, j] - retv[i, j] != 0:
                retv[i, j] = 0

    if wv.shape[1] != retv.shape[1]:
        
        raise ValueError("Input Variable Size not equal!")

    # 리밸런싱 시점 위치 정보 가져오기
    cdef np.ndarray k
    k = np.array([])

    for i in np.arange(len(weights)):
        if weights.index[i] in wtemp.index:
            k = np.append(k, i)

    for i in np.arange(len(wv)):
        if i == 0:
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > 0:
                    retv[i, j] = (1 + retv[i, j]) * (1 - bfee) - 1.
                else:
                    retv[i, j] = (1 + retv[i, j]) - 1.

        elif i != 0 and (i not in k):
            wv[i] = wv[i - 1] * (1 + retv[i - 1])
            wv[i] /= wv[i].sum()
            retv[i] = (1 + retv[i]) - 1.

        elif i != 0 and (i in k):
            for j in np.arange(wv.shape[1]):
                if wv[i, j] > wv[i - 1, j]:
                    retv[i, j] = (1 + retv[i, j]) * (1 - bfee) - 1.
                elif wv[i, j] < wv[i -1, j]:
                    retv[i, j] = (1 + retv[i, j]) * (1 - sfee) - 1.
                else:
                    retv[i, j] = (1 + retv[i, j]) - 1.
    
    return wv

cpdef double portfolioTurbulence(np.ndarray returns, np.ndarray weights):
    """
        포트폴리오 구성종목들의 Turbulence Index를 계산하기 위한 함수
        returns: 포트폴리오 구성 종목들의 수익률 시계열(np.ndarray) / 마지막 row는 직전 최근 데이터!
        weights: 포트폴리오 구성 종목들의 현재 투자비중 Spot(np.ndarray)
    """
    if returns.shape[1] != 1:
        cov = np.cov(returns.transpose(), ddof=1)
        inv = np.linalg.inv(cov)
        yt = returns[-1]
        mu = returns.transpose().mean(1)
        portTur = (weights * (yt - mu)).dot(inv).dot(yt - mu)
    else:
        cov = np.var(returns, ddof=1)
        yt = returns[-1]
        mu = returns.mean()
        portTur = (yt - mu) ** 2 / cov
    
    return portTur

def normalscore(factor, ascend=False):
    """
        factor 시계열을 받아서 시기마다 백점만점 순위 계산해서 반환
    """

    sfactor = np.zeros(factor.shape)
    for i in np.arange(len(factor)):
        ftemp = factor.iloc[i]
        sfactor[i] = ftemp.rank(ascending=not ascend)/len(ftemp) * 100
    
    sfactor = df(sfactor, index=factor.index, columns=factor.columns)
    
    return sfactor

def max_dd(data):
    """
        Maximum Drawdown을 계산하는 함수
        data: 시계열 데이터프레임(컬럼은 여러개여도 됨 다 계산해줄게)
        근데 data는 수익률같은 값이 아니라 누적이나 지수값이어야 함 알지?
    """
    
    maxv = data.cummax()
    dd = (maxv - data)/maxv
    
    mdd = df(dd, index=data.index, columns=data.columns)
    
    return -mdd.max()

def logfip(returns, p = 250):
    """
       The Frog-in-the-pan signal(FIP) signal
       Momentum 지표 중의 하나로서, 투자자들이 덜 빈번하고 극적인 변화보다
       빈번하고 점진적인 변화에 덜 민감하다는 것에 기반한 시그널
       returns: 로그수익률 데이터프레임
       p = rolling window
    """
    
    cr = returns.rolling(p).sum()
    pos = (returns > 0).astype(np.int32)
    neg = (returns < 0).astype(np.int32)
    pn = (pos.rolling(p).sum() - neg.rolling(p).sum())/p * 100.
    fip = np.exp(cr) * pn
    fip = fip.dropna()
    
    return fip

def fip(returns, p = 250):
    """
       The Frog-in-the-pan signal(FIP) signal
       Momentum 지표 중의 하나로서, 투자자들이 덜 빈번하고 극적인 변화보다
       빈번하고 점진적인 변화에 덜 민감하다는 것에 기반한 시그널
       returns: 절대수익률 데이터프레임
       p = rolling window
    """
    
    cr = (1 + returns).rolling(p).prod() - 1.
    pos = (returns > 0).astype(np.int32)
    neg = (returns < 0).astype(np.int32)
    pn = (pos.rolling(p).sum() - neg.rolling(p).sum())/p * 100.
    fip = (1 + cr) * pn
    fip = fip.dropna()
    
    return fip
    
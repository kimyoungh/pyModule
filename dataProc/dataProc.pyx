# 강화학습 기반 K200 추종 ETF 전략에 사용할 데이터 작업, 제너레이터 작성 모듈
# State 데이터 단위 구성: [공분산행렬, 상관계수 행렬, 기대수익률 대각행렬]
import pandas as pd
from pandas import DataFrame as df
import numpy as np
cimport numpy as np
from numpy import random
import quantM as qnt
import pytsignal as pt

# 공분산행렬, 상관계수행렬, 기대수익률 대각행렬 입력받아서 단위 State 데이터로 다시 정리
# 절대값이 가장 큰 값 기준으로 -1 ~ 1 사이 값으로 정규화
def sDataproc(cov, corr, ert):
    """
        cov: Covariance Matrix DataFrame with (len(Time Series) * n) * n size
        corr: Correlation Matrix DataFrame with (len(Time Series) * n) * n size
        ert: Expected Return Diagonal Matrix DataFrame with len(Time Series) * n size
        return: sdata // np.ndarray(len(Time Series) * n * n * 3) cov, corr, ert 순
    """
    
    # 같은 길이의 데이터인지, 같은 구성 데이터로 이뤄져있는지 검사 아니면 에러
    assert(cov.index[0][0] == corr.index[0][0] == ert.index[0]) # 같은 날짜로 시작하는지 검사
    assert(len(cov) == len(corr)) # 공분산행렬, 상관계수행렬 같은 시계열 길이인지 검사
    assert(len(corr)/cov.shape[1] == len(ert)) # 상관계수행렬, 기대수익률 행렬 같은 길이인지 검사
    assert(cov.index[-1][0] == corr.index[-1][0] == ert.index[-1]) # 같은 날짜로 끝나는지 검사
    assert(np.sum(cov.columns == corr.columns) == np.sum(corr.columns == cov.columns)) # 같은 구성원을 갖고 있는지 검사
    
    covalue = cov.values
    corvalue = corr.values
    ervalue = ert.values
    
    erm = np.array([])
    # 기대수익률 대각행렬로 변환
    for i in np.arange(len(ervalue)):
        em = np.diag(ervalue[i])
        erm = np.append(erm, em)
    erm = erm.reshape(cov.shape)
    
    # reshaping
    covalue = covalue.reshape(len(ert), cov.shape[1], cov.shape[1])
    corvalue = corvalue.reshape(len(ert), corr.shape[1], corr.shape[1])
    erm = erm.reshape(len(ert), ert.shape[1], ert.shape[1])
   
    # 상관계수행렬 대각성분 정리(0으로)
    temp = np.diag([-1] * corr.shape[1])
    for i in np.arange(len(corvalue)):
        corvalue[i] += temp

    # 절대값이 가장 큰 값 기준으로 -1 ~ 1 사이 값으로 정규화
    for i in np.arange(len(erm)):
        # 공분산행렬
        cmax = abs(covalue[i]).max()
        covalue[i] /= cmax
        
        # 상관계수행렬
        cmax = abs(corvalue[i]).max()
        corvalue[i] /= cmax
        
        # 기대수익률행렬
        cmax = abs(erm[i]).max()
        erm[i] /= cmax
        
    # State 데이터로 모두 합치기
    sdata = np.zeros((len(erm), cov.shape[1], cov.shape[1], 3))
    for i in np.arange(len(erm)):
        k = np.array([])
        k = np.append(k, covalue[i])
        k = np.append(k, corvalue[i])
        k = np.append(k, erm[i])
        k = k.reshape(3, cov.shape[1], cov.shape[1])
        k = k.transpose()
        sdata[i] = k
    
    return sdata

# Pixel Dropout 작업 함수(면적 기준)
cpdef np.ndarray rAreaGenerator(np.ndarray data, int smax = 2):
    """
        State 데이터의 일부 성분의 값을 임의로 0으로 만들어 데이터에 변화도 주고,
        입력 데이터 개수도 늘리는 목적의 함수(면적 기준)
        data: State를 표현하는 np.ndarray(len(Time Series), n, n, m)
        smax: 0의 값을 넣을 범위의 최대 길이(기본값: 2)
        return: rdata // state 데이터에 변화를 준 데이터
    """
    cdef int ww = data.shape[2]
    cdef int ll = data.shape[1]
    cdef int i
    cdef int j
    cdef int s
    cdef int l
    cdef int w
    cdef int xp
    cdef int yp
    cdef np.ndarray rdata
    rdata = data.copy()
    for i in np.arange(len(data)):
        for j in np.arange(data.shape[3]):
            # 0으로 만들 면적 크기 선정
            s = random.randint(0, smax + 1)
            if s != 0:
                l = random.randint(1, s+1) # 세로
                w = random.randint(1, s+1) # 가로
                xp = random.randint(1, ww - w)
                yp = random.randint(1, ll - l)
                rdata[i, yp:(yp+l), xp:(xp+w), j] = 0.
            else:
                pass
            
    return rdata

# Pixel Dropout 작업 함수(점 기준)
cpdef np.ndarray rPointGenerator(np.ndarray data, int pmax = 10):
    """
        State 데이터의 일부 성분의 값을 임의로 0으로 만들어 데이터에 변화도 주고,
        입력 데이터 개수도 늘리는 목적의 함수(점 기준)
        data: State를 표현하는 np.ndarray(len(Time Series, n, n, m)
        pmax: 0의 값을 넣을 성분의 최대 개수(기본값: 10)
        return: rdata // state 데이터에 변화를 준 데이터
    """
    cdef int ww = data.shape[2]
    cdef int ll = data.shape[1]
    cdef int i
    cdef int j
    cdef int k
    cdef int s
    cdef np.ndarray ax
    cdef np.ndarray xp
    cdef np.ndarray yp
    cdef int sz = data.shape[1] * data.shape[2]
    cdef np.ndarray rdata
    rdata = data.copy()
    for i in np.arange(len(data)):
        for j in np.arange(data.shape[3]):
            # 0으로 만들 성분 개수 선정
            s = random.randint(0, pmax + 1)
            if s != 0:
                ax = np.random.randint(0, sz, s+1)
                yp = np.floor(ax / ww).astype(np.int32)
                xp = (ax % ww).astype(np.int32) - 1
                for k in np.arange(len(xp)):
                    if xp[k] == -1:
                        xp[k] = ww - 1
                        yp[k] -= 1
                for k in np.arange(len(yp)):
                    rdata[i, yp[k], xp[k], j] = 0.
            
    return rdata

# 시계열 임의 시점 및 길이 도출 함수
cpdef np.ndarray tseriesGen(np.ndarray tseries, int min_period=250):
    """
        Time Series 데이터를 받고, 시계열 최소 길이를 받아서
        임의의 시점에 최소 길이 이상의 플레이 시계열 도출해서 반환
        tseries: 시계열을 담고 있는 numpy.ndarray 데이터
        min_period: 최소 시계열 길이
        return => 임의 시계열 인덱스
    """
    cdef np.ndarray tdata
    cdef int n = len(tseries)

    assert(n >= min_period)
    
    # t: 시작 시점
    cdef int t = np.random.randint(0, n - min_period + 1)
    cdef int p = np.random.randint(min_period, n - t + 1)

    return np.arange(t, t + p)

# 종가기준 수익률, 매도 수익률, 매수 수익률 계산
def returnsCal(popen, price):
    """
        장시작때 시가에 매매를 가정한 수익률을 계산해서 도출
        popen: 시가 시계열 데이터프레임
        price: 종가 시계열 데이터프레임
        반환값(로그수익률)=>
        returns: 종가기준 수익률
        sreturns: 매도 수익률
        breturns: 매수 수익률 반환
    """
    assert(len(popen) == len(price))

    returns = np.log(price / price.shift(1))
    sreturns = np.log(popen / price.shift(1))
    breturns = np.log(price / popen)

    return returns, sreturns, breturns

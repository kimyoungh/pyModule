# quant Machine Learning Module

import numpy as np
import pandas as pd
cimport numpy as np
from pandas import DataFrame as df
from keras.utils import np_utils
from keras import layers, models


# Simple Deep Neural Network Class for classification
class sqcDNN(models.Sequential):
    
    """
        Nin: DNN 입력 사이즈
        Nh_l: Hidden Layer 노드 수, Nh_l의 크기는 Hidden Layer 개수
        drp_l: Hidden Layer에서 Dropout 비율 리스트 / Nh_l과 같은 크기여야 하고, Dropout 적용 안하면 값 0
        Nout: 출력 사이즈
    """
    
    def __init__(self, Nin, Nh_l, drp_l, Nout, activation='relu', loss='categorical_crossentropy', optimizer='adam', metrics='accuracy'):
        super().__init__()
        self.__act = activation
        
        if len(Nh_l) != len(drp_l):
            raise ValueError("Hidden Layer와 Dropout 개수가 안맞습니다.")
        
        self.add(layers.Dense(Nh_l[0], activation=self.__act, input_shape=(Nin,)))
        if drp_l[0] != 0.:
            self.add(layers.Dropout(drp_l[0]))
        for i in np.arange(len(Nh_l)-1):
            self.add(layers.Dense(Nh_l[i + 1], activation=self.__act))
            if drp_l[i + 1] != 0.:
                self.add(layers.Dropout(drp_l[i + 1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        
        self.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        

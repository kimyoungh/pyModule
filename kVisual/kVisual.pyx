# -*- coding: utf-8 -*-
"""
@author: yk.kim
Module of Python Visualization especially for financial and pandas user
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame as df


# pd.DataFrame or pd.Series 데이터의 라인 차트
def pdLine(data, interval=250, color={}, leg_loc=0):
	"""
		Line chart for pd.DataFrame or pd.Series data
		data: DataFrame or Series
		interval: Interval of the ticker to be represented on the X_axis(default:250)
		color: dict. Line color for each element({name: color}, default: System's choice)
		leg_loc: Location of legend(default: 0(Best))
	"""
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	lt = len(data)
	xr = np.arange(lt)
	xticks = [interval * i for i in np.arange(int(lt/interval))]
	xticklabels = data.index.values[xticks]
	if type(data) == pd.DataFrame:
		cols = data.columns.values
		for i in np.arange(len(cols)):
			try:
				ax.plot(xr, data[cols[i]].values, color=color[cols[i]])
			except:
				ax.plot(xr, data[cols[i]].values)
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)
		ax.legend(cols, loc=leg_loc)
	elif type(data) == pd.Series:
		try:
			ax.plot(xr, data.values, color=color[data.name])
		except:
			ax.plot(xr, data.values)
		ax.set_xticks(xticks)
		ax.set_xticklabels(xticklabels)
		ax.legend([data.name], loc=leg_loc)
	plt.show()

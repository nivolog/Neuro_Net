import numpy as np
import pandas as pd 
import os
import sys


def sorts(experiments):
	experiments = [int(item) for item in experiments]
	experiments = sorted(experiments)
	experiments = [str(item) for item in experiments]
	return(experiments)


def testdata(a, path = './data/train/'):
	
	experiments = sorts(os.listdir(path))
	exp = path + experiments[a] + '/'
	print(exp)
	tables = sorted(os.listdir(path +  experiments[a]))
	
	 
	acc = pd.read_csv(exp + tables[0], index_col=0)
	gyro = pd.read_csv(exp + tables[1], index_col=0)
	trajectory = pd.read_csv(exp + tables[2], index_col=0)
	acc.columns = ['xa', 'ya', 'za', 'ta']
	gyro.columns = ['xg', 'yg', 'zg', 'tg']
	data = pd.concat([acc, gyro, trajectory], axis=1)
	test_X=data[['xa', 'ya', 'za', 'xg', 'yg', 'zg']]
	test_Y=data[['x', 'y']]
	
	return test_X, test_Y


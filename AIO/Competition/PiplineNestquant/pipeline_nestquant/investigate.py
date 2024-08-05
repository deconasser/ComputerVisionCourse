import glob
import polars as pl
import os
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)

# dfs = []
# for csv in glob.glob(r'.\runs\cases\*\*.csv'):
# 	df = pl.read_csv(csv)
# 	if int(csv.split(os.sep)[-2].replace('case', '')) > 18: continue
# 	df = df.with_columns(pl.lit(csv.split(os.sep)[-2].replace('case', '')).alias('case'))
# 	df = df.with_columns(pl.col('SMAPE').cast(pl.Utf8))
# 	dfs.append(df)
# dfs = pl.concat(dfs)

# print(dfs.sort(by='MAE')[0])
# print(dfs.sort(by='R2')[-1])
# print(dfs.filter(pl.col('Name') == 'ExtremeGradientBoostingRegression').sort(by='MAE', descending=True))
# dfs.filter(pl.col('Name') == 'ExtremeGradientBoostingRegression').sort(by='MAE', descending=True).write_csv('best_XGBoost.csv')
# print(dfs.filter(pl.col('Name') == 'build_Baseline_ave').sort(by='MAE', descending=True))

import json
import numpy as np
import itertools
from utils.general import yaml_load

def report():
	data = []
	# a = [['R'], ['No', 'Yes', 'YesTrain'], [5, 15], [5, 12, 20]]
	# data.extend(list(itertools.product(*a)))
	# a = [['C'], ['No'], [5, 15], [5, 12, 20]]
	# data.extend(list(itertools.product(*a)))
	# d = []
	b = []
	for i in glob.glob(r'.\runs\cases\*'):
		if '.zip' in i: continue
		# case = data[int(i.split(os.sep)[-1].replace('case', ''))]
		# case = f'l{case[3]}-g{case[2]}-{case[1]}-{case[0]}'

		if os.path.exists(os.path.join(i, 'values\invalid.npy')): n = len(np.load(os.path.join(i, 'values\invalid.npy')))
		else: n = len(np.load(os.path.join(f'runs\cases\case{int(i.split(os.sep)[-1].replace("case", "")) - 6}', 'values\invalid.npy')))

		df = pl.read_csv(os.path.join(i, 'results.csv')).filter(~pl.col("Name").str.contains("Baseline"))
		# df = pl.read_csv(os.path.join(i, 'results.csv')).filter(pl.col('Name') == 'ExtremeGradientBoostingRegression')
		# df = df.with_columns(pl.lit(case).alias('Des'))
		df = df.with_columns(pl.lit(str(i.split(os.sep)[-1].replace('case', ''))).alias('Case'))
		df = df.with_columns(pl.col('SMAPE').cast(pl.Float64))
		df = df.with_columns(pl.lit(n).alias('Num'))
		df = df.with_columns(pl.lit([str(len(pl.read_csv(f'{i}\\logs\\{n}.csv'))) if os.path.exists(f'{i}\\logs\\{n}.csv') else '_' for n in df['Name'].to_list()]).alias('Epochs'))
		opt = yaml_load(f'{i}\\updated_opt.yaml')
		df = df.with_columns(pl.lit(opt['lag']).alias('Lag'))
		df = df.with_columns(pl.lit(opt['resample ']).alias('Granularity'))
		# print(); exit()
		# exit()
		# df = df.with_columns(pl.col('Time').str.split('m')).with_columns(pl.col('Time').arr.get(0) * 60 if pl.col('Time').arr.count() == 0 else pl.col('Time').arr.get(0))
		# print(df)
		# exit()
		# df = df.drop('Name')
		b.append(df)
	print(pl.concat(b).select(['Case', 'Num', 'Name', 'Time', 'Epochs', 'Lag', 'Granularity', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2']))
	# pl.concat(b).select(['Case', 'Num', 'Time', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'R2']).write_csv('best_XGBoost_2.csv')

def est_time():
	est = []
	for path in glob.glob(r'.\runs\cases\*'):
		if '.zip' in path: continue
		if int(path.split(os.sep)[-1].replace('case', '')) < 25: continue
		if int(path.split(os.sep)[-1].replace('case', '')) > 32: continue
		samples = len(np.load(os.path.join(path, r'values\ytrain.npy')))

		def convert_to_seconds(time_str):
		    if 'm' in time_str:
		        minutes, seconds = time_str.split('m')
		        minutes = int(minutes)
		        seconds = int(seconds.rstrip('s'))
		        total_seconds = minutes * 60 + seconds
		    else:
		        total_seconds = int(time_str.rstrip('s'))
		    return total_seconds


		time = convert_to_seconds(pl.read_csv(os.path.join(path, r'results.csv')).filter(pl.col('Name') == 'ExtremeGradientBoostingRegression')['Time'].to_list()[0])

		est.append(time / samples)

	print(est)
	print(np.array(est).mean())
report()
import os
import gc
import json
import time
import random
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse
from multiprocessing.pool import ThreadPool
# from npy_append_array import NpyAppendArray
from utils.npy_utils import NpyFileAppend

from utils.general import yaml_load
from utils.general import list_convert
from utils.general import flatten_list

from utils.option import model_dict

from rich.progress import track
from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn

# from sklearn.preprocessing import MinMaxScaler

class DatasetController():
    def __init__(self, 
                 configsPath=None, 
                 resample=5, 
                 # startTimeId=0, 
                 workers=8, 
                 splitRatio=(0.7, 0.2, 0.1), 
                 lag=5, 
                 ahead=1, 
                 offset=1, 
                 savePath='.', 
                 polarsFilling=None, 
                 machineFilling=None,
                 low_memory=False,
                 normalization=False):
        """ Read data config """
        self.dataConfigs = yaml_load(configsPath)

        try:
            self.dataPaths = self.dataConfigs['data']
            self.dateFeature = self.dataConfigs['date']
            self.timeID = self.dataConfigs['time_id']
            self.targetFeatures = self.dataConfigs['target']
            self.delimiter = self.dataConfigs['delimiter']
            self.trainFeatures = list_convert(self.dataConfigs['features'])
            self.dirAsFeature = self.dataConfigs['dir_as_feature']
            self.splitDirFeature = self.dataConfigs['split_dir_feature']
            self.splitFeature = self.dataConfigs['split_feature']
            self.timeFormat = self.dataConfigs['time_format']
            self.granularity = self.dataConfigs['granularity']
            self.startTimeId = self.dataConfigs['start_time_id']

            self.yearStart   = self.dataConfigs['year_start']
            self.yearEnd     = self.dataConfigs['year_end']
            self.monthStart  = self.dataConfigs['month_start']
            self.monthEnd    = self.dataConfigs['month_end']
            self.dayStart    = self.dataConfigs['day_start']
            self.dayEnd      = self.dataConfigs['day_end']
            self.hourStart   = self.dataConfigs['hour_start']
            self.hourEnd     = self.dataConfigs['hour_end']
            self.minuteStart = self.dataConfigs['minute_start']
            self.minuteEnd   = self.dataConfigs['minute_end']
            
            self.X_train = []
            self.y_train = []
            self.X_val = []
            self.y_val = []
            self.X_test = []
            self.y_test = []
        except KeyError:
            # print('Getting data from files')
            self.X_train = np.load(file=self.dataConfigs['X_train'], mmap_mode='r')
            self.y_train = np.load(file=self.dataConfigs['y_train'], mmap_mode='r')
            self.X_val = np.load(file=self.dataConfigs['X_val'], mmap_mode='r')
            self.y_val = np.load(file=self.dataConfigs['y_val'], mmap_mode='r')
            self.X_test = np.load(file=self.dataConfigs['X_test'], mmap_mode='r')
            self.y_test = np.load(file=self.dataConfigs['y_test'], mmap_mode='r')

            # print(f'{self.X_train.shape = }')
            # print(f'{self.y_train.shape = }')
            # print(f'{self.X_val.shape = }')
            # print(f'{self.y_val.shape = }')
            # print(f'{self.X_test.shape = }')
            # print(f'{self.y_test.shape = }')
            # print('Done', end='\n\n')

        self.configsPath = configsPath
        self.workers = workers
        self.resample = resample
        # self.startTimeId = startTimeId
        self.splitRatio = splitRatio
        self.lag = lag
        self.ahead = ahead
        self.offset = offset
        self.savePath = savePath
        self.polarsFilling = polarsFilling
        self.machineFilling = machineFilling

        self.df = None
        self.dataFilePaths = []
        self.dirFeatures = []
        self.segmentFeature = None

        self.num_samples = []
        self.low_memory = low_memory
        self.smoothing = False
        self.normalization = normalization
        self.scaler = None

    def execute(self, cyclicalPattern=False):
        if len(self.y_train) == 0:
            self.GetDataPaths(dataPaths=self.dataPaths)
            self.ReadFileAddFetures(csvs=self.dataFilePaths, dirAsFeature=self.dirAsFeature, hasHeader=True)
            self.df = self.df.with_columns(self.df['OPEN_TIME'].cast(pl.Datetime).alias('OPEN_TIME'))
            dictionary = {"1INCHUSDT": 1, "AAVEUSDT": 2, "ADAUSDT": 3}
            self.df = self.df.with_columns(self.df['SYMBOL'].apply(lambda x: dictionary.get(x)).alias("ID"))
            # print(self.df)
            # exit()
            self.df = self.df.drop_nulls()
            self.df = self.df.unique()
            

            # =============================================================================
            # self.segmentFeature = 'kml_segment_id'
            # save_dir = Path(self.savePath) / 'values'
            # save_dir.mkdir(parents=True, exist_ok=True)
            # u = self.df[self.segmentFeature].unique()
            # with self.ProgressBar() as progress:
            #     for ele in progress.track(u, description='Splitting jobs'):
            #         d = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
            #         d = d.with_columns(pl.col(self.dateFeature).cast(pl.Date))
            #         d.write_csv(file=save_dir / f'{ele}.csv', separator=self.delimiter)
            # exit()
            # =============================================================================

            if self.timeFormat is not None: self.UpdateDateColumnDataType(dateFormat=self.timeFormat, f=pl.Datetime, t=pl.Datetime)
            self.TimeIDToDateTime(timeIDColumn=self.timeID, granularity=self.granularity, startTimeId=self.startTimeId)
            self.df = self.StripDataset(df=self.df)
            self.GetSegmentFeature(dirAsFeature=self.dirAsFeature, splitDirFeature=self.splitDirFeature, splitFeature=self.splitFeature)

            #self.df = self.df.unique(subset=[self.segmentFeature, self.dateFeature, self.timeID], maintain_order=True)


            self.df = self.GetUsedColumn(df=self.df)
            # if self.machineFilling: self.MachineLearningFillingMissingData(model=self.machineFilling)
            # if self.polarsFilling: self.PolarsFillingMissingData(strategy=self.polarsFilling, low_memory=self.low_memory)
            # if cyclicalPattern: self.CyclicalPattern()
            self.df = self.StripDataset(df=self.df)

            self.df = self.ReduceMemoryUsage(df=self.df, info=True)
            # with self.ProgressBar() as progress:
            #     for i in progress.track(range(10), description='Adding columns'):
            #         self.df = self.df.with_columns(pl.col(self.targetFeatures).alias(f'thecol{i}'))
            #         # self.df = self.df.shrink_to_fit()
            #         self.trainFeatures.append(f'thecol{i}')

            if self.smoothing:
                self.df = self.df.with_columns(pl.col(self.targetFeatures).rolling_mean(window_size=5))
                self.df = self.df.drop_nulls()
            # print(self.df); exit()
            # self.trainFeatures.append('roll')
            # print(self.df
            self.SplittingData(splitRatio=self.splitRatio, lag=self.lag, ahead=self.ahead, offset=self.offset, multimodels=False, low_memory=self.low_memory)      
        return self

    def SortDataset(self):
        if self.segmentFeature:
            if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
            else: self.df = self.df.sort(by=[self.segmentFeature])

    def StripDataset(self, df):
        # print(df)
        # if pandas: df = pl.from_pandas(data=df.reset_index())
        # print(df)
        if self.yearStart  : df = df.filter(pl.col(self.dateFeature).dt.year()   >= self.yearStart)
        if self.yearEnd    : df = df.filter(pl.col(self.dateFeature).dt.year()   < self.yearEnd)
        if self.monthStart : df = df.filter(pl.col(self.dateFeature).dt.month()  >= self.monthStart)
        if self.monthEnd   : df = df.filter(pl.col(self.dateFeature).dt.month()  < self.monthEnd)
        if self.dayStart   : df = df.filter(pl.col(self.dateFeature).dt.day()    >= self.dayStart)
        if self.dayEnd     : df = df.filter(pl.col(self.dateFeature).dt.day()    < self.dayEnd)
        if self.hourStart  : df = df.filter(pl.col(self.dateFeature).dt.hour()   >= self.hourStart)
        if self.hourEnd    : df = df.filter(pl.col(self.dateFeature).dt.hour()   < self.hourEnd)
        if self.minuteStart: df = df.filter(pl.col(self.dateFeature).dt.minute() >= self.minuteStart)
        if self.minuteEnd  : df = df.filter(pl.col(self.dateFeature).dt.minute() < self.minuteEnd)
        # if pandas: df = df.to_pandas()
        return df

    def MachineLearningFillingMissingData(self, model):
        self.SortDataset()
        self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime))
        if self.segmentFeature:
            with self.ProgressBar() as progress:
                dfs = None
                for ele in progress.track(self.df[self.segmentFeature].unique(), description='  Filling data'):
                    df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    df = self.FillDate(df=df)
                    df = df.with_columns(pl.col(self.segmentFeature).fill_null(pl.lit(ele)))
                    if dfs is None: dfs = df
                    else: dfs = pl.concat([dfs, df])
                self.df = dfs
        else: 
            self.df = self.FillDate(df=self.df)
        self.CyclicalPattern()
        self.df = self.GetUsedColumn(df=self.df)
        null_or_not = (self.df.null_count() > 0).rows(named=True)[0]
        target = [key for key, value in null_or_not.items() if value]
        independence = [key for key, value in null_or_not.items() if not value]
        independence.remove(self.dateFeature)
        for t in target:
            with_null = self.df.filter(pl.col(t).is_null()).drop(self.dateFeature)
            without_null = self.df.filter(pl.col(t).is_not_null()).drop(self.dateFeature)
            for item in model_dict:
                if item['type'] == 'MachineLearning':
                    for flag in [item['model'].__name__, *item['alias']]:
                        if model == flag:
                            model = item['model'](modelConfigs=item['config'], save_dir=None)
                            model.modelConfigs['verbosity'] = 0 
                            model.build()
                            model.fit(X_train=without_null[independence].to_numpy(),
                                      y_train=without_null[target].to_numpy())
                            with_null = with_null.with_columns(pl.lit(model.predict(with_null[independence].to_numpy())).alias(t))
                            self.df = self.df.join(other=with_null, on=independence, how="left", suffix="_right")\
                                             .select([
                                                pl.when(pl.col(f'{t}_right').is_not_null())
                                                .then(pl.col(f'{t}_right'))
                                                .otherwise(pl.col(t))
                                                .alias(t),
                                                *independence,
                                                self.dateFeature
                                             ])
                            break
                    else: continue  # only executed if the inner loop did NOT break
                    break  # only executed if the inner loop DID break
    
    def PolarsFillingMissingData(self, strategy, low_memory=False):
        self.SortDataset()
        if self.segmentFeature:
            if low_memory:    
                p = Path(self.savePath , "temp")
                p.mkdir(parents=True, exist_ok=True)
            else: dfs = None
            with self.ProgressBar() as progress:
                for ele in progress.track(self.df[self.segmentFeature].unique(), description='  Filling data'):
                    df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    df = self.FillDate(df=df)
                    df = df.with_columns(pl.col(self.segmentFeature).fill_null(pl.lit(ele)))
                    for f in [feature for feature in [*self.trainFeatures, self.targetFeatures] if feature != self.segmentFeature]:
                        df = df.with_columns(pl.col(f).fill_null(strategy=strategy))
                    if low_memory: df.write_csv(file=p / f'{ele}.csv', separator=self.delimiter)
                    else:
                        if dfs is None: dfs = df
                        else: dfs = pl.concat([dfs, df])

            if low_memory:
                self.df = None
                self.dataFilePaths = []
                self.dataPaths = []
                self.GetDataPaths(p)
                self.ReadFileAddFetures(csvs=self.dataFilePaths, dirAsFeature=0, hasHeader=True, description='ReReading data')
                # add self.df = self.df.shrink_to_fit()
                # import shutil
                #             shutil.rmtree(p)
            else: self.df = dfs
        else:    
            self.df = self.FillDate(df=self.df)
            for f in [feature for feature in [*self.trainFeatures, self.targetFeatures] if feature != self.segmentFeature]:
                self.df = df.with_columns(pl.col(f).fill_null(strategy=strategy))

    def GetData(self, shuffle):
        # if self.low_memory:
        #     if shuffle:
        #         random.shuffle(self.xtr)
        #         random.shuffle(self.ytr)
        #         random.shuffle(self.xv)
        #         random.shuffle(self.yv)
        #         random.shuffle(self.xt)
        #         random.shuffle(self.yt)
        #     self.X_train = self.xtr
        #     self.y_train = self.ytr
        #     self.X_val = self.xv
        #     self.y_val = self.yv
        #     self.X_test = self.xt
        #     self.y_test = self.yt
        # else:
        if shuffle:
            np.random.shuffle(self.X_train)
            np.random.shuffle(self.y_train)
            np.random.shuffle(self.X_val)
            np.random.shuffle(self.y_val)
            np.random.shuffle(self.X_test)
            np.random.shuffle(self.y_test)
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def GetDataPaths(self, dataPaths=None, extensions=('.csv')):
        if dataPaths: self.dataPaths = dataPaths
        if not isinstance(self.dataPaths, list): self.dataPaths = [self.dataPaths]
        for i in self.dataPaths:
            # print(i, os.path.exists(i)) 
            if os.path.isdir(i):
                for root, dirs, files in os.walk(i):
                    for file in files:
                        if file.endswith(extensions): 
                            self.dataFilePaths.append(os.path.join(root, file))
            elif i.endswith(extensions) and os.path.exists(i): 
                self.dataFilePaths.append(i)
        assert len(self.dataFilePaths) > 0, 'No csv file(s)'
        # t = [4844]
        # t = [5583,8602,9235,7107,8062,600,599,499,5989,8604,8318,3302,8065,2715,4879,2636,942,4166,5402,8256,5296,6829,9270,9064,3237,9231,3806,7877,8683,9233,7593,9234,6226,1256,6228,8822,3651,5988,6828,7696,7148,6666,7259,7320,1257,3303,3665,3805,8300,3650,6225,8948,7321,8947,6619,8319,6997,48,7875,8063,7694,4878,8697,8821,5150,446,2635,7258,4165,7594,9272,47,1258,3666,308,4831,8497,4269,8603,7878,3664,8061,5582,6667,5404,6665,6826,974,7695,7109,8498,2561,8900,309,445,5405,5990,6227,6827,8605,9096,9271,8901,9269,8255,2637,7876,5403,8902,9063,6618,49,8317,3238,8824,4732,9236,6350,7257,6617,4731,6406,7595,4270,2560,7140,4927,7108,8946,4272,4167,9353,9062,941,8682,8066,4271,4168,4832,4834,6996,8606,8823,8064,9232,1989,4928,6349,765,975,4833,9352]
        # self.dataFilePaths = [os.path.abspath(csv) for csv in list_convert(self.dataFilePaths) if int(os.path.basename(csv).split('.')[0]) in t]
        self.dataFilePaths = [os.path.abspath(csv) for csv in list_convert(self.dataFilePaths)]
        # import random
        # random.shuffle(self.dataFilePaths)
        # self.dataFilePaths = self.dataFilePaths[:150]
        # [print(p.replace(r'E:\01.Code\00.Github\UTSF-Reborn', '.')) for p in self.dataFilePaths]
        # exit()

    def ProgressBar(self):
        return Progress("[bright_cyan][progress.description]{task.description}",
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("•Items"),
                        MofNCompleteColumn(), # "{task.completed}/{task.total}",
                        TextColumn("•Remaining"),
                        TimeRemainingColumn(),
                        TextColumn("•Total"),
                        TimeElapsedColumn())

    def ReduceMemoryUsage(self, df, info=False):
        before = round(df.estimated_size('gb'), 4)
        Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
        Numeric_Float_types = [pl.Float32,pl.Float64]    
        for col in df.columns:
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            elif col_type == pl.Utf8:
                df = df.with_columns(df[col].cast(pl.Categorical))
            else:
                pass
        if info: print(f"Memory usage: {before} GB => {round(df.estimated_size('gb'), 4)} GB")
        return df


    def ReadFileAddFetures(self, csvs=None, dirAsFeature=0, newColumnName='dir', hasHeader=True, description='  Reading data'):
        if csvs: self.dataFilePaths = [os.path.abspath(csv) for csv in csvs]  
        if dirAsFeature != 0: self.dirAsFeature = dirAsFeature

        if self.dirAsFeature == 0:
            # df = pl.read_csv(source=self.dataFilePaths, separator=self.delimiter, has_header=hasHeader, try_parse_dates=True, low_memory=True)
            with self.ProgressBar() as progress:
                d = []
                for csv in progress.track(self.dataFilePaths, description=description):
                    data = pl.read_csv(source=csv, separator=self.delimiter, has_header=hasHeader, try_parse_dates=True, low_memory=True, infer_schema_length=10_000)
                    if 'rain' in data.columns: data = data.with_columns(pl.col('rain').cast(pl.Float64))
                    d.append(data)
                df = pl.concat(d)
        else:
            dfs = []
            for csv in track(self.dataFilePaths, description=description):
                features = [int(p) if p.isdigit() else p for p in csv.split(os.sep)[-self.dirAsFeature-1:-1]]
                df = pl.read_csv(source=csv, separator=self.delimiter, has_header=hasHeader, try_parse_dates=True)
                for idx, f in enumerate(features): 
                    df = df.with_columns(pl.lit(f).alias(f'{newColumnName}{idx}'))
                self.dirFeatures.append(f'{newColumnName}{idx}')
                df = df.shrink_to_fit()
                dfs.append(df)
            df = pl.concat(dfs)
            self.dirFeatures = list(set(self.dirFeatures))
            self.trainFeatures.extend(self.dirFeatures)
    
        if self.df is None: self.df = df
        else: self.df = pl.concat([self.df, df])

        # if self.dateFeature: self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime))

    def TimeIDToDateTime(self, timeIDColumn=None, granularity=1, startTimeId=0):
        if timeIDColumn: self.timeID = timeIDColumn
        if granularity != 1: self.granularity = granularity
        if startTimeId != 0: self.startTimeId = startTimeId

        if not self.timeID: return 

        max_time_id = self.df[self.timeID].max() * self.granularity + self.startTimeId - 24*60
        assert max_time_id <= 0, f'time id max should be {(24*60 - self.startTimeId) / self.granularity} else it will exceed to the next day'

        # print(self.df)
        # self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime).alias('c'))
        self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime) + pl.duration(minutes=(pl.col(self.timeID)-1)*self.granularity+self.startTimeId))
        # self.df = self.df.with_columns(pl.duration(minutes=(pl.col(self.timeID)-1)*self.granularity+self.startTimeId).alias('m'))
        # self.df = self.df.sort(self.dateFeature); print(self.df); exit()

    def GetUsedColumn(self, df, exclude_date=False):
        if exclude_date: alist = [self.trainFeatures, self.targetFeatures]
        else: alist = [self.dateFeature, self.trainFeatures, self.targetFeatures] 
        # else: alist = [self.dateFeature, self.trainFeatures, self.targetFeatures, 'time_id'] 
        return df[[col for i in alist for col in (i if isinstance(i, list) else [i])]]

    def UpdateDateColumnDataType(self, f=pl.Datetime, t=pl.Datetime, dateFormat='%Y-%M-%d'):
        self.df = self.df.with_columns(pl.col(self.dateFeature).str.strptime(f, fmt=dateFormat).cast(t))

    def GetSegmentFeature(self, dirAsFeature=0, splitDirFeature=0, splitFeature=None):
        if dirAsFeature != 0: self.dirAsFeature = dirAsFeature
        if splitDirFeature != 0: self.splitDirFeature = splitDirFeature
        if splitFeature is not None: self.splitFeature = splitFeature

        assert not all([self.dirAsFeature != 0, self.splitFeature is not None])
        self.segmentFeature = self.dirFeatures[self.splitDirFeature] if self.dirAsFeature != 0 and self.splitDirFeature != -1 else self.splitFeature if self.splitFeature else None
        # TODO: consider if data in segmentFeature are number or not. 

    def TimeEncoder(self, df):
        day = 24 * 60 * 60 # Seconds in day  
        year = (365.2425) * day # Seconds in year

        # df = self.FillDate(df=df)
        unix = df[self.dateFeature].to_frame().with_columns(pl.col(self.dateFeature).cast(pl.Utf8).alias('unix_str'))
        unix_time = [time.mktime(parse(t).timetuple()) for t in unix['unix_str'].to_list()]
        df = df.with_columns(pl.lit(unix_time).alias('unix_time'))

        if len(set(df[self.dateFeature].dt.day().to_list())) > 1:
            df = df.with_columns(np.cos((pl.col('unix_time')) * (2 * np.pi / day)).alias('day_cos'))
            df = df.with_columns(np.sin((pl.col('unix_time')) * (2 * np.pi / day)).alias('day_sin'))
            self.trainFeatures.extend(['day_cos', 'day_sin'])
        if len(set(df[self.dateFeature].dt.month().to_list())) > 1:
            df = df.with_columns(np.cos((pl.col('unix_time')) * (2 * np.pi / year)).alias('month_cos'))
            df = df.with_columns(np.sin((pl.col('unix_time')) * (2 * np.pi / year)).alias('month_sin'))
            self.trainFeatures.extend(['month_cos', 'month_sin'])
        
        return df

    def CyclicalPattern(self):
        # assert self.dateFeature is not None
        # if self.segmentFeature:
        #     if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
        #     else: self.df = self.df.sort(by=[self.segmentFeature])
        
        # if self.segmentFeature:
        #     dfs = None
        #     for ele in self.df[self.segmentFeature].unique():
        #         df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
        #         df = self.TimeEncoder(df=df)
        #         if dfs is None: dfs = df
        #         else: dfs = pl.concat([dfs, df])
        #     self.df = dfs.drop_nulls()
        # else: 
        #     self.df = self.TimeEncoder(df=self.df).drop_nulls()
        # self.df = self.TimeEncoder(df=self.df)
        # pass
        self.df = self.df.with_columns(
                    pl.col(self.dateFeature).dt.weekday().alias('weekday'),
                    pl.col(self.dateFeature).dt.month().alias('month'),
                    ((pl.col(self.dateFeature).dt.hour()*60 + pl.col(self.dateFeature).dt.minute())/self.resample).alias(f'{self.resample}mins_hour'),
                )
        self.df = self.df.with_columns(
            ((pl.col(f'{self.resample}mins_hour')/(24*(60/self.resample))*2*np.pi).sin()).alias('hour_sin'), # 0 to 23 -> 23h55
            ((pl.col(f'{self.resample}mins_hour')/(24*(60/self.resample))*2*np.pi).cos()).alias('hour_cos'), # 0 to 23 -> 23h55
            ((pl.col('weekday')/(7)*2*np.pi).sin()).alias('day_sin'), # 1 - 7
            ((pl.col('weekday')/(7)*2*np.pi).cos()).alias('day_cos'), # 1 -  7
            ((pl.col('month')/(12)*2*np.pi).sin()).alias('month_sin'), # 1 -12
            ((pl.col('month')/(12)*2*np.pi).cos()).alias('month_cos') # 1-12
        )
        self.trainFeatures.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'])
        

    def FillDate(self, df=None, low=None, high=None, granularity=None, pandas=False): 
        if not self.dateFeature: return
        if df is None: df = self.df
        if not low: low=df[self.dateFeature].min()
        if not high: high=df[self.dateFeature].max()
        if not granularity: granularity=self.granularity

        if pandas: 
            df = pl.from_pandas(data=df.reset_index(drop=True))
            # df = df.drop('index')
            df = df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime).dt.cast_time_unit("us"))
        # print(type(df) == pd.D)

        # print(df)

        d = pl.date_range(low=low,
                          high=high,
                          interval=timedelta(minutes=granularity),
                          closed='both',
                          name=self.dateFeature).to_frame()

        # print(d)

        df = df.join(other=d, 
                     on=self.dateFeature, 
                     how='outer')
        # print(df)
        df = self.StripDataset(df=df)
        # print(df)

        if pandas: 
            df = df.to_pandas().reset_index(drop=True)
        return df

    def TimeBasedCrossValidation(self, args):
        d, lag, ahead, offset, splitRatio, progressBar = args
        features = []
        labels = []
        if not progressBar:
            for idx in range(len(d)-offset-lag+1):
                feature = d[idx:idx+lag]
                label = d[self.targetFeatures][idx+lag+offset-ahead:idx+lag+offset].to_frame()
                # print(d[idx:idx+lag])
                # print(d[idx+lag+offset-ahead:idx+lag+offset])
                # print('==================================')
                if all(flatten_list(feature.with_columns(pl.all().is_not_null()).rows())) and all(flatten_list(label.with_columns(pl.all().is_not_null()).rows())): 
                    labels.append(np.squeeze(label.to_numpy()))
                    features.append(feature.to_numpy()) 
                    # print(d[idx:idx+lag])
                    # print(d[idx+lag+offset-ahead:idx+lag+offset])
                    # print('==================================')

            # for frame in d.iter_slices(n_rows=lag+offset):
            #     print(frame[:lag])
            #     print(frame[lag+offset-ahead:lag+offset])
            # thething()
        else:
            with self.ProgressBar() as progress:
                for idx in progress.track(range(len(d)-offset-lag+1), description='Splitting data'):
                    feature = d[idx:idx+lag]
                    label = d[self.targetFeatures][idx+lag+offset-ahead:idx+lag+offset].to_frame()
                    # print(d[idx:idx+lag])
                    # print(d[idx+lag+offset-ahead:idx+lag+offset])
                    # print('==================================')
                    if all(flatten_list(feature.with_columns(pl.all().is_not_null()).rows())) and all(flatten_list(label.with_columns(pl.all().is_not_null()).rows())): 
                        labels.append(np.squeeze(label.to_numpy()))
                        features.append(feature.to_numpy()) 

        length = len(features)
        if splitRatio[1]==0 and splitRatio[2]==0: 
            train_end = length 
            val_end = length
        elif splitRatio[1]!=0 and splitRatio[2]==0:
            train_end = int(length*splitRatio[0])
            val_end = length
        else:
            train_end = int(length*splitRatio[0])
            val_end = int(length*(splitRatio[0] + splitRatio[1]))
        return [features[0:train_end], features[train_end:val_end], features[val_end:length]],\
               [  labels[0:train_end],   labels[train_end:val_end],   labels[val_end:length]]

    def SplittingData(self, splitRatio, lag, ahead, offset, multimodels=False, low_memory=False, save_dir=None):
        if not save_dir: save_dir = self.savePath
        save_dir = Path(save_dir) / 'values'
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.segmentFeature:
            if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
            else: self.df = self.df.sort(by=[self.segmentFeature])
        
        if offset<ahead: offset=ahead

        if low_memory:
            xtr = save_dir / 'xtrain.npy'
            ytr = save_dir / 'ytrain.npy'
            xv  = save_dir / 'xval.npy'
            yv  = save_dir / 'yval.npy'
            xt  = save_dir / 'xtest.npy'
            yt  = save_dir / 'ytest.npy'
            xtr_writer =  NpyFileAppend(xtr)
            ytr_writer =  NpyFileAppend(ytr)
            xv_writer =  NpyFileAppend(xv)
            yv_writer =  NpyFileAppend(yv)
            xt_writer =  NpyFileAppend(xt)
            yt_writer =  NpyFileAppend(yt)
            # for p in [path, xtr, xv, xt, ytr, yv, yt]: p.mkdir(parents=True, exist_ok=True)

        # for ele in self.df[self.segmentFeature].unique():
        #     d = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
        #     d = self.FillDate(df=d)
        #     self.TimeBasedCrossValidation([d, lag, ahead, offset, splitRatio, False])
        #     exit()

        if self.segmentFeature:
            data = []
            u = self.df[self.segmentFeature].unique()
            # u = [5583,8602,9235,7107,8062,600,599,499,5989,8604,8318,3302,8065,2715,4879,2636,942,4166,5402,8256,5296,6829,9270,9064,3237,9231,3806,7877,8683,9233,7593,9234,6226,1256,6228,8822,3651,5988,6828,7696,7148,6666,7259,7320,1257,3303,3665,3805,8300,3650,6225,8948,7321,8947,6619,8319,6997,48,7875,8063,7694,4878,8697,8821,5150,446,2635,7258,4165,7594,9272,47,1258,3666,308,4831,8497,4269,8603,7878,3664,8061,5582,6667,5404,6665,6826,974,7695,7109,8498,2561,8900,309,445,5405,5990,6227,6827,8605,9096,9271,8901,9269,8255,2637,7876,5403,8902,9063,6618,49,8317,3238,8824,4732,9236,6350,7257,6617,4731,6406,7595,4270,2560,7140,4927,7108,8946,4272,4167,9353,9062,941,8682,8066,4271,4168,4832,4834,6996,8606,8823,8064,9232,1989,4928,6349,765,975,4833,9352]
            with self.ProgressBar() as progress:
                for ele in progress.track(u, description='Splitting jobs'):
                    d = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    d = self.FillDate(df=d)
                    d.drop_in_place(self.dateFeature)
                    data.append([d, lag, ahead, offset, splitRatio, False])
            
            if self.df is not None: self.df.write_csv(save_dir / 'data_processed.csv')
            del self.df
            gc.collect()

            with self.ProgressBar() as progress:
                task_id = progress.add_task("Splitting data", total=len(data))
                with ThreadPool(self.workers) as p:
                    for idx, result in enumerate(p.imap(self.TimeBasedCrossValidation, data)):
                        x = result[0]
                        y = result[1]

                        if low_memory:
                            # np.save(open(xtr / f'{u[idx]}.npy', 'wb'), x[0])
                            # np.save(open(ytr / f'{u[idx]}.npy', 'wb'), y[0])
                            # np.save(open(xv / f'{u[idx]}.npy', 'wb'), x[1])
                            # np.save(open(yv / f'{u[idx]}.npy', 'wb'), y[1])
                            # np.save(open(xt / f'{u[idx]}.npy', 'wb'), x[2])
                            # np.save(open(yt / f'{u[idx]}.npy', 'wb'), y[2])
                            # print(f'{len(y[0]) = }')
                            # print(f'{len(y[1]) = }')
                            # print(f'{len(y[2]) = }')
                            # print('====================================')

                            if all([len(y[0]) > 0,
                                    len(y[1]) > 0,
                                    len(y[2]) > 0]):
                            # try:
                                xtr_writer.append(np.array(x[0]))
                                ytr_writer.append(np.array(y[0]))
                                xv_writer.append(np.array(x[1]))
                                yv_writer.append(np.array(y[1]))
                                xt_writer.append(np.array(x[2]))
                                yt_writer.append(np.array(y[2]))
                            # except ValueError:
                            #     print(f'{np.array(x[0]) = }')
                            #     print(f'{np.array(y[0]) = }')
                            #     print(f'{np.array(x[1]) = }')
                            #     print(f'{np.array(y[1]) = }')
                            #     print(f'{np.array(x[2]) = }')
                            #     print(f'{np.array(y[2]) = }')
                            #     print('====================================')
                        else:
                            if multimodels:
                                self.X_train.append(x[0])
                                self.y_train.append(y[0])
                                self.X_val.append(x[1])
                                self.y_val.append(y[1])
                                self.X_test.append(x[2])
                                self.y_test.append(y[2])
                            else:
                                self.X_train.extend(x[0])
                                self.y_train.extend(y[0])
                                self.X_val.extend(x[1])
                                self.y_val.extend(y[1])
                                self.X_test.extend(x[2])
                                self.y_test.extend(y[2])
                        
                        self.num_samples.append({'id' : u[idx],
                                                'train': len(y[0]),
                                                'val': len(y[1]),
                                                'test': len(y[2])})
                        progress.advance(task_id)
            if low_memory:
                xtr_writer.close()
                xv_writer.close()
                xt_writer.close()
                ytr_writer.close()
                yv_writer.close()
                yt_writer.close()
        else:
            assert not self.low_memory, 'Low memory mode: to be implemented'
            if self.df is not None: self.df.write_csv(save_dir / 'data_processed.csv')
            d = self.df.clone()
            del self.df
            gc.collect()
            d = self.FillDate(df=d)
            d.drop_in_place(self.dateFeature) 
            
            x, y = self.TimeBasedCrossValidation(args=[d, lag, ahead, offset, splitRatio, True]) 
            self.X_train.extend(x[0])
            self.y_train.extend(y[0])
            self.X_val.extend(x[1])
            self.y_val.extend(y[1])
            self.X_test.extend(x[2])
            self.y_test.extend(y[2])
            self.num_samples.append({'train': len(y[0]),
                                     'val': len(y[1]),
                                     'test': len(y[2])})


        # save_dir = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\values')
        # xtr = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\splitted\xtrain')
        # ytr = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\splitted\ytrain')
        # xv = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\splitted\xval')
        # yv = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\splitted\yval')
        # xt = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\splitted\xtest')
        # yt = Path(r'E:\01.Code\00.Github\UTSF-Reborn\runs\exp191\splitted\ytest')
        # self.num_samples = {'train': 1,
        #                              'val': 2,
        #                              'test': 3}

        if low_memory:
            # self.xtr = [Path(xtr) / f for f in os.listdir(xtr) if f.endswith('.npy')]
            # self.ytr = [Path(ytr) / f for f in os.listdir(ytr) if f.endswith('.npy')]
            # self.xv = [Path(xv) / f for f in os.listdir(xv) if f.endswith('.npy')]
            # self.yv = [Path(yv) / f for f in os.listdir(yv) if f.endswith('.npy')]
            # self.xt = [Path(xt) / f for f in os.listdir(xt) if f.endswith('.npy')]
            # self.yt = [Path(yt) / f for f in os.listdir(yt) if f.endswith('.npy')]
            
            self.X_train = np.load(file=xtr, mmap_mode='r')
            self.y_train = np.load(file=ytr, mmap_mode='r')
            self.X_val = np.load(file=xv, mmap_mode='r')
            self.y_val = np.load(file=yv, mmap_mode='r')
            self.X_test = np.load(file=xt, mmap_mode='r')
            self.y_test = np.load(file=yt, mmap_mode='r')

            # self.y_train = self.LoadNumpyDir(ytr)
            # np.save(open(save_dir / 'y_train.npy', 'wb'), self.y_train)
            # self.y_val = self.LoadNumpyDir(yv)
            # np.save(open(save_dir / 'y_val.npy', 'wb'), self.y_val)
            # self.y_test = self.LoadNumpyDir(yt)
            # np.save(open(save_dir / 'y_test.npy', 'wb'), self.y_test)
            # self.X_test = self.LoadNumpyDir(xt)
            # np.save(open(save_dir / 'X_test.npy', 'wb'), self.X_test)
            # self.X_val = self.LoadNumpyDir(xv)
            # np.save(open(save_dir / 'X_val.npy', 'wb'), self.X_val)
            # self.X_train = self.LoadNumpyDir(xtr)
            # np.save(open(save_dir / 'X_train.npy', 'wb'), self.X_train)
        else:
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)
            self.X_val = np.array(self.X_val)
            self.y_val = np.array(self.y_val)
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)
            if len(self.X_train) != 0: np.save(open(save_dir / 'X_train.npy', 'wb'), self.X_train)
            if len(self.y_train) != 0: np.save(open(save_dir / 'y_train.npy', 'wb'), self.y_train)
            if len(self.X_val) != 0: np.save(open(save_dir / 'X_val.npy', 'wb'), self.X_val)
            if len(self.y_val) != 0: np.save(open(save_dir / 'y_val.npy', 'wb'), self.y_val)
            if len(self.X_test) != 0: np.save(open(save_dir / 'X_test.npy', 'wb'), self.X_test)
            if len(self.y_test) != 0: np.save(open(save_dir / 'y_test.npy', 'wb'), self.y_test)

        if len(self.num_samples) != 0: 
            with open(save_dir / "num_samples.json", "w") as final: 
                json.dump(self.num_samples, final, indent=4) 


    def SplittingData(self, splitRatio, lag, ahead, offset, multimodels=False, low_memory=False, save_dir=None):
        if not save_dir: save_dir = self.savePath
        save_dir = Path(save_dir) / 'values'
        save_dir.mkdir(parents=True, exist_ok=True)

        inp_len = lag
        lag_window = inp_len + 1
        id_col_name = self.segmentFeature
        digit = self.resample
        sample_time = f'{digit}min'
        pd_time = 'm'
        pl.Config.set_tbl_rows(100)
        pl.Config.set_tbl_cols(100)

        self.SortDataset()
        if self.df is not None: 
            self.df.write_csv(save_dir / 'data_processed.csv')
            del self.df 
            gc.collect()
        pl_dataframe = pl.read_csv(save_dir / 'data_processed.csv', try_parse_dates=True, low_memory=self.low_memory)
        pl_dataframe = self.ReduceMemoryUsage(df=pl_dataframe, info=True)

        LIST_FEATURE = [self.targetFeatures]
        #if self.CyclicalPattern: LIST_FEATURE[:0] = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

        FINAL_LIST_FEATURE = []
        for ft_l in LIST_FEATURE:
            # if ft_l == self.segmentFeature:
            #     FINAL_LIST_FEATURE.append(ft_l)
            #     continue

            for i in range(0, lag_window):
                if i == 0:
                    FINAL_LIST_FEATURE.append(ft_l)
                else:
                    FINAL_LIST_FEATURE.append(ft_l+f'_lag_{lag_window-i}')

        current_geopath_id_count_df = sorted(int(i) for i in pl_dataframe[id_col_name].unique())
        
        xtr = save_dir / 'xtrain.npy'
        ytr = save_dir / 'ytrain.npy'
        xv  = save_dir / 'xval.npy'
        yv  = save_dir / 'yval.npy'
        xt  = save_dir / 'xtest.npy'
        yt  = save_dir / 'ytest.npy'
        xtr_writer =  NpyFileAppend(xtr)
        ytr_writer =  NpyFileAppend(ytr)
        xv_writer =  NpyFileAppend(xv)
        yv_writer =  NpyFileAppend(yv)
        xt_writer =  NpyFileAppend(xt)
        yt_writer =  NpyFileAppend(yt)

        invalid = []

        with self.ProgressBar() as progress:
            for id_ in progress.track(current_geopath_id_count_df, description='Splitting data'):
                x_train = []
                y_train = []
                x_val = []
                y_val = []
                x_test = []
                y_test = []
                sub_pl_dataframe = pl_dataframe.filter(pl.col(id_col_name)==id_)

                #########################################################
                # Resample 
                pd_df = sub_pl_dataframe.to_pandas()
                pd_df = pd_df.drop(['ID'], axis=1)
                inserted_row = [pd_df.iloc[0][self.dateFeature].floor(sample_time)]
                # print(inserted_row)
                # exit()

                if inserted_row[0] != pd_df.iloc[0][self.dateFeature]:
                    inserted_row.extend(list(np.full(shape=(len(pd_df.columns)-1), fill_value=np.nan)))
                    pd_df = pd.concat([pd.DataFrame([inserted_row], columns=pd_df.columns), pd_df])
                pd_df = pd_df.set_index(self.dateFeature).asfreq(sample_time)
                pd_df.reset_index(inplace=True)
                
                sub_pl_dataframe = sub_pl_dataframe.groupby_dynamic(self.dateFeature, every=f"{digit}{pd_time}").agg(pl.col(self.targetFeatures).mean())
                sub_pl_dataframe = sub_pl_dataframe.to_pandas()
                sub_pl_dataframe = sub_pl_dataframe.set_index(self.dateFeature).asfreq(sample_time)
                sub_pl_dataframe.reset_index(inplace=True)
                pd_df = pd_df.combine_first(sub_pl_dataframe)



                #########################################################
                # Feature engienerin
                
                # =======================================================================================
                pd_df = pl.from_pandas(pd_df)
                pd_df = pd_df[LIST_FEATURE + [self.dateFeature]].to_pandas()
                # print(pd_df)
                pd_df = pd_df.set_index(self.dateFeature)
                # print(pd_df)
                # exit()
                # =======================================================================================

                for ft in LIST_FEATURE:
                    for i in range(1, lag_window):
                        pd_df = pd_df.merge(pd_df[ft].shift(i, freq=sample_time),
                            left_index=True,
                            right_index=True,
                            how='left',
                            suffixes=('', f'_lag_{i}'))
               
                pd_df = pd_df.dropna()
                pd_df = pd_df[FINAL_LIST_FEATURE]
                
                # print(pd_df)
                # exit()

                input_feature =  pd_df.drop(columns=LIST_FEATURE).to_numpy()
                
                target_feature = pd_df[self.targetFeatures].to_numpy()
                # if debug:
                #     print(pd_df.drop(columns=LIST_FEATURE).iloc[0])
                
                num_samples = len(target_feature)
                train_num = train_idx = int(num_samples*splitRatio[0])
                # print(f'train')
                val_id = int(num_samples*(splitRatio[0] + splitRatio[1]))
                val_num = val_id-train_num
                test_num = num_samples-(train_num+val_num)
                
                if any([input_feature[0:train_idx].size == 0, 
                        target_feature[0:train_idx].size == 0, 
                        input_feature[train_idx:val_id].size == 0, 
                        target_feature[train_idx:val_id].size == 0, 
                        input_feature[val_id:].size == 0, 
                        target_feature[val_id:].size == 0]):
                    # print(f'{id_} is empty')
                    invalid.append(id_)
                    continue

                x_train.extend(np.swapaxes(input_feature[0:train_idx].reshape((train_num, -1, inp_len)), 1,2 ))
                y_train.extend(target_feature[0:train_idx] )
                x_val.extend(np.swapaxes(input_feature[train_idx:val_id].reshape((val_num, -1, inp_len)), 1,2 ) )
                y_val.extend(target_feature[train_idx:val_id] )
                x_test.extend(np.swapaxes(input_feature[val_id:].reshape((test_num, -1, inp_len)), 1,2 ) )
                y_test.extend(target_feature[val_id:])


                x_train = np.array(x_train)
                y_train = np.array(y_train)
                x_val = np.array(x_val)
                y_val = np.array(y_val)
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                
                # Insert ID
                x_train = np.concatenate([np.full_like(x_train[..., 0:1], fill_value=id_), x_train], axis=-1)
                x_val = np.concatenate([np.full_like(x_val[...,0:1], fill_value=id_), x_val], axis=-1)
                x_test = np.concatenate([np.full_like(x_test[...,0:1], fill_value=id_), x_test], axis=-1)
                # print(x_train)
                # exit()
                xtr_writer.append(x_train)
                ytr_writer.append(y_train)
                xv_writer.append(x_val)
                yv_writer.append(y_val)
                xt_writer.append(x_test)
                yt_writer.append(y_test)
                
                assert np.count_nonzero(np.isnan(x_test)) == 0, 'There are at least one null value in test set'
                assert np.count_nonzero(np.isnan(x_val)) == 0, 'There are at least one null value in val set'
                assert np.count_nonzero(np.isnan(x_train)) == 0, 'There are at least one null value in train set'
                assert np.count_nonzero(np.isnan(y_test)) == 0, 'There are at least one null value in test set'
                assert np.count_nonzero(np.isnan(y_val)) == 0, 'There are at least one null value in val set'
                assert np.count_nonzero(np.isnan(y_train)) == 0, 'There are at least one null value in train set'

            self.X_train = np.load(file=xtr, mmap_mode='r+')
            self.y_train = np.load(file=ytr, mmap_mode='r+')
            self.X_val = np.load(file=xv, mmap_mode='r')
            self.y_val = np.load(file=yv, mmap_mode='r')
            self.X_test = np.load(file=xt, mmap_mode='r')
            self.y_test = np.load(file=yt, mmap_mode='r')

            if self.normalization:
                self.scaler = {'max' : self.X_train[:, : , -1].max(), 
                               'min' : self.X_train[:, : , -1].min(),
                               # 'max_rain' : self.X_train[:, : , -2].max(), 
                               # 'min_rain' : self.X_train[:, : , -2].min()
                               }
                # print(self.X_train[:3, : , -1])
                # print(self.y_train[:3], end='\n\n\n\n')
                self.X_train[:, : , -1] = (self.X_train[:, : , -1] - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])
                # self.X_train[:, : , -2] = (self.X_train[:, : , -2] - self.scaler['min_rain']) / (self.scaler['max_rain'] - self.scaler['min_rain'])
                self.y_train = (self.y_train - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])
                x = self.X_train[:, : , -1] * (self.scaler['max'] - self.scaler['min']) +  self.scaler['min']
                y = self.y_train * (self.scaler['max'] - self.scaler['min']) +  self.scaler['min']

                # print(x[:3])
                # print(y[:3])
                # print(f'{self.X_train[:, : , -1].max() = }')
                # print(f'{self.X_train[:, : , -1].min() = }')
                # print(f'{self.X_train[:, : , -2].max() = }')
                # print(f'{self.X_train[:, : , -2].min() = }')
                # print(f'{self.y_train.max() = }')
                # print(f'{self.y_train.min() = }')
                # exit()

                # self.scaler = {'max' : self.X_train[:, : , -1].max(), 'min' : self.X_train[:, : , -1].min()}
                # self.X_train[:, : , -1] = (self.X_train[:, : , -1] - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])
                # self.y_train = (self.y_train - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])

                # print(self.y_train.max(), self.y_train.min())
                # print(self.scaler)
                # json.dump(self.scaler, open(save_dir / 'scaler.save', 'w'), indent=4, separators=(',', ': ')) 

            np.save(save_dir / 'invalid.npy', invalid)

        
    
    def LoadNumpyDir(self, dir_path):
        files = [Path(dir_path) / f for f in os.listdir(dir_path) if f.endswith('.npy')]
        files.sort()

        data = None
        # with self.ProgressBar() as progress:
        #     task_id = progress.add_task("  Reading data", total=len(files))
        #     with ThreadPool(self.workers) as p:
        #         for result in p.imap(lambda file: np.load(file=file, mmap_mode='r'), files):
        #             if data is not None: data = np.concatenate([data, result], axis=0)
        #             else: data = result
        #             progress.advance(task_id)

        with self.ProgressBar() as progress:
            for f in progress.track(files, description='  Reading data'):
                arr = np.load(file=f, mmap_mode='r')
                if data is not None: data = np.concatenate((data, arr), axis=0)
                else: data = arr
        return data

    def display(self): pass

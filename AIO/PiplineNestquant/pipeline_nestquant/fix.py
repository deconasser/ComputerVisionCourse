import glob
import polars as pl

directory = r'.\data\raw\traffic\a_year-by_id-new'
delimiter = '|'
hasHeader = True
dateFeature = 'date'

for csv in glob.glob(directory + "/*.csv"):
    df = pl.read_csv(source=csv, separator=delimiter, has_header=hasHeader, try_parse_dates=True)
    df = df.with_columns(pl.col(dateFeature).cast(pl.Date))
    df = df.sort([dateFeature, 'time_id'])
    df.write_csv(file=csv, separator=delimiter, has_header=hasHeader)
    # print(df)
    # exit()
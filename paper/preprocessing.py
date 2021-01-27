import pandas as pd
import os
import numpy as np
import itertools
import h5py
from sklearn import preprocessing
from scipy import signal
import pickle
months = {
    1: 'jan',
    2: 'feb',
    3: 'march',
    4: 'april',
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'aug',
    9: 'sept',
    10: 'oct',
    11: 'nov',
    12: 'dec'
}


def flatten_series(data, lats, lons, time_type, num_years, min_year):
    liltemp = data.T
    num_lats = len(lats)
    num_lons = len(lons)
    num_times = data.shape[0]

    flat_df = pd.DataFrame([i for i in itertools.product(lats, lons)], columns=['lat', 'lon'])
    flat_df = pd.concat([flat_df, pd.DataFrame(liltemp.reshape(num_lats * num_lons, num_times))], axis=1)
    flat_df = flat_df.melt(id_vars=['lat', 'lon'])
    flat_df.columns = ['lat', 'lon', 'time', 'temp']
    chunks = flat_df.time.nunique() / num_years
    flat_df['year'] = flat_df.time.apply(lambda x: x // chunks + min_year)
    flat_df['time'] = flat_df.time.apply(lambda x: int(x % chunks))
    if time_type == 'monthly':
        flat_df['month'] = flat_df.time.apply(lambda x: months[x + 1])
    elif time_type == 'weekly':
        flat_df['month'] = flat_df.time.apply(lambda x: month_from_week(x + 1, month_weeks))
        # deal with overlapping months (this will be slow and dumb)
        bb2 = flat_df[flat_df.time.isin([x - 1 for x in overlap])]
        bb2['month'] = bb2.month.apply(lambda x: x.split(', ')[0])
        flat_df.loc[flat_df.time.isin([x - 1 for x in overlap]), 'month'] = flat_df.loc[
            flat_df.time.isin([x - 1 for x in overlap]), 'month'].apply(lambda x: x.split(', ')[1])
        flat_df = flat_df.append(bb2)
    flat_df.lat = flat_df.lat.astype(int)
    flat_df.lon = flat_df.lon.astype(int)
    flat_df.year = flat_df.year.astype(int)
    flat_df.time = flat_df.time.astype(int)
    return flat_df


def flatten_lens(data, lats, lons, min_year):
    liltemp = data.T
    num_lats = len(lats)
    num_lons = len(lons)
    num_times = data.shape[0]
    num_years = int(num_times/12)
    min_year = 1920

    flat_df = pd.DataFrame([i for i in itertools.product(lats, lons)], columns=['lat', 'lon'])
    flat_df = pd.concat([flat_df, pd.DataFrame(liltemp.reshape(num_lats * num_lons, num_times))], axis=1)
    flat_df = flat_df.melt(id_vars=['lat', 'lon'])
    flat_df.columns = ['lat', 'lon', 'time', 'val']
    chunks = flat_df.time.nunique() / num_years
    flat_df['year'] = flat_df.time.apply(lambda x: x // chunks + min_year)
    flat_df['time'] = flat_df.time.apply(lambda x: int(x % chunks))
    flat_df['month'] = flat_df.time.apply(lambda x: months[x + 1])

    flat_df.lat = flat_df.lat.astype(int)
    flat_df.lon = flat_df.lon.astype(int)
    flat_df.year = flat_df.year.astype(int)
    flat_df.time = flat_df.time.astype(int)
    return flat_df


def monthly_X(df, months=['july', 'aug', 'sept', 'oct'], step=1, agg=False,
              min_year=1940, max_year=2014, scale=True):
    """
    @param df:
    @param months:
    @param step:
    @param agg:
    @param min_year:
    @param max_year:
    @param scale:
    @return:
    """
    df = df.loc[(df.year >= min_year) & (df.year <= max_year) & (df.month.isin(months))]
    if 'land_mask' in df.columns:
        df = df.loc[df.land_mask > 0]
    df = df.dropna()
    if step > 1:
        if not agg:
            df = df.loc[(df.lat % step == 0) & (df.lon % step == 0)]
        else:
            to_bin = lambda x: np.floor(x / step) * step + step / 2
            df.loc[:, "lat"] = df.lat.map(to_bin)
            df.loc[:, "lon"] = df.lon.map(to_bin)
    mdf = df.groupby(['lat', 'lon', 'year', 'month']).val.mean().reset_index()
    X = np.vstack(mdf.groupby('year').val.apply(np.array))

    if scale:
        X = preprocessing.scale(X)

    fts = mdf.drop_duplicates(['lat', 'lon', 'month'])
    # NZI/ENSO flags
    fts.loc[:, 'nzi'] = 0
    fts.loc[:, 'enso'] = 0
    fts.loc[(fts.lat.isin(range(-40, -26))) & (fts.lon.isin(range(170, 201))), 'nzi'] = 1
    fts.loc[(fts.lat.isin(range(-5, 5))) & (fts.lon.isin(range(160, 210))), 'enso'] = 3
    fts.loc[(fts.lat.isin(range(-5, 5))) & (fts.lon.isin(range(210, 270))), 'enso'] = 4
    fts.loc[(fts.lat.isin(range(-5, 5))) & (fts.lon.isin(range(190, 240))), 'enso'] = 3.4
    fts = fts.reset_index().drop('index', axis=1)
    return X, fts[['lat', 'lon', 'month', 'nzi', 'enso']], mdf


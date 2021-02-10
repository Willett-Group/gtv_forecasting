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

    #flat_df.lat = flat_df.lat.astype(int)
    #flat_df.lon = flat_df.lon.astype(int)
    flat_df.year = flat_df.year.astype(int)
    flat_df.time = flat_df.time.astype(int)
    return flat_df


def detrend_and_scale(df, min_year, max_year):
    # min_ and max_year inclusive
    sst = df[(df.year>=min_year)&(df.year<=max_year)]
    seasonal = df.groupby(['lat', 'lon', 'month']).val.agg(['mean', 'std']).reset_index()
    sst = pd.merge(sst, seasonal, on=['lat', 'lon', 'month'])
    sst['anomaly'] = (sst['val'] - sst['mean'])/sst['std']
    sst['val'] = sst['anomaly']
    # aggregate to 10x10 temporal scale, stack into matrix, scale
    X, fts, _ = monthly_X(sst, step=10, agg=True, scale=False, max_year=2018)
    X = preprocessing.scale(signal.detrend(X, axis=0))
    return X, fts


def monthly_X(df, months=['july', 'aug', 'sept', 'oct'], step=1, agg=False,
              min_year=1940, max_year=2019, scale=True):
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


def load_precipitation(all_divs=False):
    df = pd.DataFrame()
    for a, b, files in os.walk('data/new_precip/'):
        for f in files:
            if 'prec' in f:
                tmp = pd.read_csv('data/new_precip/' + f,
                                    header= None,
                                    names= ['date', 'precip', 'diff_from_mean'])
                tmp['file'] = '_'.join(f.split('_')[1:3])
                df = df.append(tmp)
    # we only keep period 03/1941- 03/2015
    df = df[(df.date>=194103)]
    areas = pd.read_csv('data/new_precip/ClDiv_areas.txt', header=None, sep='.')
    areas = areas.drop(11, axis=1)
    areas['st'] = areas[0].apply(lambda x: x.split()[0])
    areas[0] = areas[0].apply(lambda x: x.split()[1])
    cols = 'DIV1   DIV2   DIV3   DIV4   DIV5   DIV6   DIV7   DIV8   DIV9  DIV10 STATE_AREA ST'.lower().split()
    areas.columns = cols
    # according to https://www.esrl.noaa.gov/psd/data/usclimdivs/descript.html, we have the following state mappings
    # california(04), arizona(02), nevada(26), utah(42)
    areas = areas[areas.st.isin(['04', '02', '26', '42'])]
    areas['state'] = ['Arizona', 'California', 'Nevada', 'Utah']
    areas = pd.melt(areas, id_vars=['state'], value_vars=['div1', 'div2', 'div3', 'div4', 'div5', 'div6', 'div7'])
    areas.columns=['state', 'division', 'area']
    areas['area'] = areas.area.astype(int)
    areas['division'] = areas.division.apply(lambda x: int(x.replace('div', '')))
    areas['file'] = areas.apply(lambda x: str(x.division)+"_"+x.state, axis=1)
    # merge data
    precip = pd.merge(df, areas[['file', 'area']], on='file')
    precip['year'] = precip.date.apply(lambda x: int(np.floor(x/100)))
    # these are the divisions that we consider for the mean precipitation amount
    # see Figure 1 and discussion in Mamalakis et al (2018) for more info
    divs2keep = [
        '4_California',
        '5_California',
        '6_California',
        '7_California',
        '1_Arizona',
        '2_Arizona',
        '3_Arizona',
        '4_Arizona',
        '5_Arizona',
        '6_Arizona',
        '7_Arizona',
        '3_Nevada',
        '4_Nevada',
        '1_Utah',
        '2_Utah',
        '4_Utah',
        '6_Utah',
        '7_Utah'
    ]
    if not all_divs:
        return precip[precip.file.isin(divs2keep)]
    else:
        return precip

def load_response():
    precip = load_precipitation()
    Y = precip.pivot('year', 'file', 'precip').values
    precip['weighted_precip'] = precip.apply(lambda x: x.precip*x.area, axis=1)
    avg_rain = (precip.groupby('date').weighted_precip.sum()/sum(precip.area.unique())).reset_index()
    y = np.array(avg_rain.weighted_precip)
    yavg = y
    regions = [x.split( '_')[1]+'({})'.format(x.split('_')[0]) for x in precip.pivot('year', 'file', 'precip').columns]
    return yavg, Y, regions



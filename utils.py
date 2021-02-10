from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error


def edge_incidence(S, threshold=0):
    """
    Computes the thresholded edge-incidence matrix of a covariance matrix for use in GTV
    :param S: covariance matrix (standardized so diagonals are all 1)
    :param threshold: float between 0 and 1 that all entries of the covariance matrix below are ignored
    :return: thresholded edge-incidence matrix
    """
    edges = np.where(abs(S)>threshold)
    edges_ix = np.where(edges[0]<edges[1])
    ix1 = edges[0][edges_ix]
    ix2 = edges[1][edges_ix]
    ix = np.arange(len(ix1))

    D = np.zeros([len(ix1), S.shape[0]])
    D[ix, ix1] = np.sqrt(abs(S[ix1, ix2]))
    D[ix, ix2] = -np.sign(S[ix1, ix2])*np.sqrt(abs(S[ix1, ix2]))
    return D

def plot_coefs(fts, color='sign', title='', month='', se=None,
               savefig=False, colorbar=False, vmin=None, vmax=None, cmap=None):
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure(figsize=(6,8))
    ax = plt.axes(projection=ccrs.Mercator(central_longitude=150, min_latitude=-60, max_latitude=60))
    ax.set_extent([80, 280, -60, 60])
    ax.stock_img()

    #nzi
    ax.add_patch(Rectangle(xy=[170, -40], width=30, height=15,
                           edgecolor='blue',
                           alpha=0.2,
                           transform=ccrs.PlateCarree())
                 )

    #nino
    for pts in [(190, 240)]:
        ur = (pts[1], 5)
        lr = (pts[0], -5)
        box = Rectangle(lr, ur[0]-lr[0], ur[1]-lr[1],
                        edgecolor='blue',
                        alpha=0.2,
                        transform=ccrs.PlateCarree())
        ax.add_patch(box)

    #pc = PatchCollection(boxes, edgecolor='blue', alpha=.1)
    #ax.add_collection(pc)

    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                  linewidth=.5, color='gray', alpha=0.5)
    # Label axes of a Plate Carree projection with a central longitude of 180:

    ax.set_xticks(list(np.arange(80, 281, 10)), crs=ccrs.PlateCarree())
    plt.xticks(rotation=45)
    ax.set_yticks(list(np.arange(-60,61,10)), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.grid(linewidth=.25)

    if month == '':
        mfts = fts
    else:
        mfts = fts[fts.month==month]
    if se is not None:
        plt.scatter(mfts.lon, mfts.lat, c=np.sign(mfts.coef), s=(abs(mfts.coef)+se) * 1000,
                    transform=ccrs.PlateCarree(), vmin=-1, vmax=1, alpha=.2)
    if color == 'sign':
        pt = plt.scatter(mfts.lon, mfts.lat, c=np.sign(mfts.coef), s=abs(mfts.coef)*1000,
                    transform=ccrs.PlateCarree(), vmin=-1, vmax=1)
    elif color == 'size':
        pt = plt.scatter(mfts.lon, mfts.lat, c=mfts.coef, cmap=cmap,
                         transform=ccrs.PlateCarree())
    if colorbar:
        if vmin is not None:
            plt.clim(vmin, vmax)
            plt.colorbar(pt, fraction=.03)
        else:
            plt.colorbar(pt, fraction=.03)
    plt.title(month+str(title))
    plt.show()


def draw_lambda_contour(df, metric, vmin=None, vmax=None):
    sns.set(font_scale=1.25)
    d = df.pivot(index='lambda_1', columns='lambda_tv', values=metric)
    if vmin is None:
        vmin = df[metric].min()
        vmax = df[metric].max()
    d = d.interpolate(method='linear', axis=1, limit_direction='both')
    d = d.sort_index(ascending=False)
    sns.heatmap(d, vmin=vmin, vmax=vmax, cmap='coolwarm')
    plt.xlabel('$\lambda_{TV}$', fontweight='bold')
    plt.ylabel('$\lambda_1$', fontweight='bold')
    #plt.xticks(np.linspace(0, d.shape[0], min(d.shape[0], 10)), rotation='vertical')
    #plt.yticks(np.linspace(0, d.shape[0], min(d.shape[0], 10)), rotation='horizontal')

def draw_lambda_contour_paper_format(df, metric, vmin=None, vmax=None):
    sns.set(font_scale=2)
    plt.figure(figsize=(10,8))
    d = df.pivot(index='lambda_1', columns='lambda_tv', values=metric)
    if vmin is None:
        vmin = df[metric].min()
        vmax = df[metric].max()
    d = d.interpolate(method='linear', axis=1, limit_direction='both')
    d = d.sort_index(ascending=False)
    d.columns = [np.round(x,2) for x in d.columns]
    ax = sns.heatmap(d, vmin=vmin, vmax=vmax, cmap='coolwarm')
    labels1 = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels1):
        if(i%2 == 0): labels1[i] = '' # skip even labels
    ax.set_xticklabels(labels1, rotation=0)
    labels2 = ax.get_yticklabels()# get x labels
    for i,l in enumerate(labels2):
        if(i%2 == 0): labels2[i] = '' # skip even labels
    ax.set_yticklabels(labels2, rotation=0)
    ax.tick_params(left=True, bottom=True)
    plt.xlabel('$\lambda_{TV}$ x $10^{-3}$')
    plt.ylabel('$\lambda_1$ x $10^{-1}$')


def threshold_covariance(S, threshold):
    p = S.shape[1]
    if threshold ==0:
        return S
    edges = np.where(abs(S) > threshold)
    edges_ix = np.where(edges[0] < edges[1])
    ix1 = edges[0][edges_ix]
    ix2 = edges[1][edges_ix]
    St = np.zeros((p, p))
    St[list(ix1), list(ix2)] = S[list(ix1), list(ix2)]
    St[list(ix2), list(ix1)] = S[list(ix2), list(ix1)]
    return St


def plot_covariance(covs, titles, threshold=0, h=4):
    n = len(covs)
    if n == 1:
        fig = plt.figure(figsize=(h + 1, h))
    else:
        fig = plt.figure(figsize=(n*h+h/2, h))
    axes=[]
    plots = []
    subplot_ix = 100 + n*10
    for i in range(n):
        ax = fig.add_subplot(subplot_ix+i+1)
        axes.append(ax)
    for i, S in enumerate(covs):
        St = threshold_covariance(S, threshold)
        g = sns.heatmap(St, ax=axes[i], vmin=-1, vmax=1, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=False)
        plots.append(g)
        plots[i].set_title(titles[i])
    mappable = plots[0].get_children()[0]
    plt.colorbar(mappable, ax = axes, orientation = 'vertical')


def shorten_region(region):
    if region in ['Area-weighted average', 'Areal Average']:
        return 'Areal \n Average'
    if len(region) > 5:
        state_map = {
            'Arizona': 'AZ',
            'Utah': 'UT',
            'Nevada': 'NV',
            'California': 'CA'
        }
        return state_map[region.split('(')[0]] + '(' + region.split('(')[1]
    else:
        return region


def plot_regional_metrics(df, metric='mse', order=None, pal=['navy', 'orange', 'green', 'firebrick'], figsize=(25,5)):
    """
    Dataframe should have the following columns:
    - Region
    - Method
    - Test MSE or R2
    """
    # make sure regions are short
    df['Region'] = df.Region.apply(lambda x: shorten_region(x))
    plt.figure(figsize=figsize)
    sns.set(font_scale=2)
    if metric == 'mse':
        sns.barplot(x='Region', y='Test MSE', hue='Method', data=df, hue_order=order)
        plt.ylabel('MSE', fontweight='bold')
    elif metric == 'r2':
        df['R2'] = df.R2.apply(lambda x: max(.01, x))
        sns.barplot(x='Region', y='R2', hue='Method', data=df, hue_order=order,
                    palette=pal)
        plt.ylabel('$R^2$', fontweight='bold')
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=4)
    plt.xlabel('Region', fontweight='bold')

def compute_errors(y_true, y_predicted):
    r2 = r2_score(y_true, y_predicted)
    mse = mean_squared_error(y_predicted, y_true)
    return(mse, r2)

def temporal_split(X, y, t):
    X_train = X[:t]
    X_test = X[t:]
    y_train = y[:t]
    y_test = y[t:]
    return X_train, X_test, y_train, y_test


def teleconnections(X, y, fts, alpha, t=50):
    tele_errors = []
    lm = linear_model.LinearRegression()

    Xnzi = X[:, fts[fts.nzi>0].index].mean(axis=1).reshape(-1,1)
    Xenso = X[:, fts[fts.enso==3.4].index].mean(axis=1).reshape(-1,1)
    Xtele = np.hstack([Xnzi, Xenso])

    if alpha == 1:
        X_train, X_test, y_train, y_test = temporal_split(Xnzi, y, t)
        lm.fit(X_train, y_train)
        tele_errors.append(['NZI',
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xenso, y, t)
        lm.fit(X_train, y_train)
        tele_errors.append(['Nino 3.4',
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xtele, y, t)
        lm.fit(X_train, y_train)
        tele_errors.append(['Nino 3.4 & NZI',
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])
    else:
        n = t
        weights = np.array([alpha**(n-t) for t in np.arange(1, n+1)])
        X_train, X_test, y_train, y_test = temporal_split(Xnzi, y, t)
        lm.fit(X_train, y_train, sample_weight=weights)
        tele_errors.append(['NZI',
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xenso, y, t)
        lm.fit(X_train, y_train, sample_weight=weights)
        tele_errors.append(['Nino 3.4',
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])

        X_train, X_test, y_train, y_test = temporal_split(Xtele, y, t)
        lm.fit(X_train, y_train, sample_weight=weights)
        tele_errors.append(['Nino 3.4 & NZI',
                            compute_errors(y_test, X_test@lm.coef_)[0],
                            compute_errors(y_test, X_test@lm.coef_)[1],
                           lm.coef_])
    df_tele = pd.DataFrame(tele_errors, columns=['Method', 'MSE', 'R2', 'Coefs'])
    df_tele['alpha'] = alpha
    return df_tele

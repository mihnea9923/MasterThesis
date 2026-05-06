# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:40:49 2023
Temperature Whiplash：

cold to hot temperature
or
hot to cold temperature

@author: sjwu

"""

#%% import libraries
import xarray as xr
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
warnings.filterwarnings("ignore")      
import pymannkendall as mk
from scipy import stats
import gc
from tqdm import tqdm
import copy
from matplotlib import ticker
import geopandas as gpd
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = 'Arial'


import regionmask
def region_mask(data):

    mask = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(data)
    data_mask = data.where((mask == 0))
    
    return data_mask




def scilimit(ax, limit_1 = -1, limit_2 = 3):

    '''
    设置坐标轴的label为科学技术法
    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    limit_1 : TYPE, optional
        DESCRIPTION. The default is -1.
    limit_2 : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    formatter : TYPE
        DESCRIPTION.

    '''
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((limit_1,limit_2)) 
    return formatter

def season_mean(da, concat=True):
    """
    Calculate seasonal mean
    """

    ## Annual mean
    da_annual = da.groupby("time.year").mean()
    
    # Seasonal mean
    # time="QS-DEC"表示以12月1日为基准, 重新取样的时间频率为每个季度的第一个月（Quarterly Start-DEC）
    da_season = da.resample(time="QS-DEC").mean()  
    
    if concat==False:
        return da_annual
    
    if concat==True:
        ## Winter: DJF
        da_djf = da_season.sel(time=(da_season["time.month"]==12))
        da_djf["time"] = da_djf["time.year"] + 1
        da_djf = da_djf.rename({"time":"year"})
     
        ## Spring: MAM
        da_mam = da_season.sel(time=(da_season["time.month"]==3))
        da_mam["time"] = da_mam["time.year"]
        da_mam = da_mam.rename({"time":"year"})
    
        ## Summer: JJA
        da_jja = da_season.sel(time=(da_season["time.month"]==6))
        da_jja["time"] = da_jja["time.year"]
        da_jja = da_jja.rename({"time":"year"})
    
        ## Autumn: SON
        da_son = da_season.sel(time=(da_season["time.month"]==9))
        da_son["time"] = da_son["time.year"]
        da_son = da_son.rename({"time":"year"})
        
        
        ## Concat all da_mean in all seasons
        da_all = xr.concat([da_annual, da_djf, da_mam, da_jja, da_son], dim="season")
        da_all["season"] = ["Annual", "DJF", "MAM", "JJA", "SON"]
    
        return  da_all

def season_sum(da, concat=True):
    """
    Calculate seasonal mean
    """
    # ds = xr.open_dataset('G:/01PhD/01-25 Temperature_whiplash_no_bias/Method_2/02_Result/01_Other_result_detrended_rolling/01_event/whiplash_event-5/std-1/HadGHCND_detrended_rolling_tas_polyfit_order-3_whiplash_event-5_1950-2014.nc')
    # ds_ = ds['C2W'].groupby('time.year').mean()
    # data_season = ds['C2W']/ds['C2W']
    
    # data_year = ds_/ds_
    # del ds, ds_
    # gc.collect()
    
    # # Annual mean
    # da_annual = da.groupby("time.year").sum()
    # # Mask
    # da_annual = land_sea_mask(da_annual, time = True, time_var = 'year', no_Antarctica = True)
    # # da_annual = da_annual * data_year
    # ## Seasonal sum
    # ##time="QS-DEC"表示以12月1日为基准, 重新取样的时间频率为每个季度的第一个月（Quarterly Start-DEC）
    # da_season = da.resample(time="QS-DEC").sum()  
    # # Mask
    # da_season = land_sea_mask(da_season, time = True, time_var = 'time', no_Antarctica = True)
    # # da_season = da_season * data_season
    
    ## Annual sum (skip NaN values during summation)
    da_annual = da.groupby("time.year").sum(min_count=1)   # Skip NaN values in annual sum

    # Seasonal sum (skip NaN values during summation)
    # time="QS-DEC" 表示重新取样的时间频率为每个季度的第一个月（Quarterly Start-DEC）
    da_season = da.resample(time="QS-DEC").sum(min_count=1)   # Skip NaN values in seasonal sum

    
    if concat==False:
        return da_annual
    
    if concat==True:
        ## Winter: DJF
        da_djf = da_season.sel(time=(da_season["time.month"]==12))
        da_djf["time"] = da_djf["time.year"] + 1
        da_djf = da_djf.rename({"time":"year"})
     
        ## Spring: MAM
        da_mam = da_season.sel(time=(da_season["time.month"]==3))
        da_mam["time"] = da_mam["time.year"]
        da_mam = da_mam.rename({"time":"year"})
    
        ## Summer: JJA
        da_jja = da_season.sel(time=(da_season["time.month"]==6))
        da_jja["time"] = da_jja["time.year"]
        da_jja = da_jja.rename({"time":"year"})
    
        ## Autumn: SON
        da_son = da_season.sel(time=(da_season["time.month"]==9))
        da_son["time"] = da_son["time.year"]
        da_son = da_son.rename({"time":"year"})
        
        ## Concat all da_mean in all seasons
        da_all = xr.concat([da_annual, da_djf, da_mam, da_jja, da_son], dim="season")
        da_all["season"] = ["Annual", "DJF", "MAM", "JJA", "SON"]
    
        return  da_all
    
    
def convert_lon(data):

    lon = (((data[get_lon_name(data)] + 180) % 360) - 180)   # 获取转变后的lon
    idx1 = np.where(lon<0)[0]
    idx2 = np.where(lon>=0)[0]
    data_convert = data.assign_coords(lon= xr.concat((lon[idx1], lon[idx2]), dim='lon'))  #重新分配lon
    data_convert.values = xr.concat((data_convert[:, idx1], data_convert[:,  idx2]), dim='lon')     
    
    return data_convert    
    del lon, idx1, idx2
    
# regional mean weighted by latitudes
def regional_mean(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_mean = data.weighted(weights).mean(("lon", "lat"))
    return regional_mean

# regional std weighted by latitudes
def regional_std(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_std = data.weighted(weights).std(("lon", "lat"))
    return regional_std


## 迭代复制数组
def iterate_array(arr, iterations, range_end):
    result = []
    for i in range(iterations):
        result.extend([x % range_end for x in arr])
    return result

def linear_trend(x, y):  
    mask = (~(np.isnan(x))) & (~(np.isnan(y)))
    
    # calculate the trend with 80% valid data
    if len(x[mask]) <= 0.2 * len(x):
        return np.nan, np.nan,np.nan, np.nan
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        result = mk.yue_wang_modification_test(y[mask])
        # result = mk.hamed_rao_modification_test(y[mask])
        p_value1 = result.p
        # return slope, p_value1, intercept, std_err
        return slope, p_value1, intercept, std_err

def get_lon_name(ds_):
    for lon_name in ['lon', 'longitude']:
        if lon_name in ds_.coords:
            return lon_name
    raise RuntimeError("Couldn't find a longitude coordinate")
    
def get_lat_name(ds_):
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds_.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")


def moving_average(data, window=15):
    threshold = data.groupby("time.dayofyear").mean()
    threshold = xr.concat([threshold, threshold, threshold], dim='dayofyear')
    threshold = threshold.rolling(center=True,  dayofyear=window).mean()
    threshold = threshold[366:732, ...]
    return threshold

def moving_std(data, window=15):
    threshold = data.groupby("time.dayofyear").std()
    threshold = xr.concat([threshold, threshold, threshold], dim='dayofyear')
    threshold = threshold.rolling(center=True,  dayofyear=window).mean()
    threshold = threshold[366:732, ...]
    return threshold




def cool2warm_whiplash(flag, max_gap=5):
    """ Identify cold2hot temperature whiplash events
    
    Parameters
    ----------
    flag : temperature flag series (1: hot, -1: cold, 0: non-hot & non-cold)
    max_gap : the maximum gap between the start and end of the event
        
    
    Returns
    -------
    cold2hot_loc: two-column arrary with the first/second column marking the start/end of the whiplash event
    """
    
    ## length of the time series
    n = len(flag)
    
    ## index location of the identified cold2hot events
    cold2hot_loc = [] 
    
    ## search for cold2hot events
    for i in range(0, n):
        if (flag[i] == -1):  # if cold extreme found
            is_cold = True
            is_hot = False

            # search for hot extreme in the following 'max_gap' days
            for j in range(i+1, min(i+max_gap+1, n)):
                if (flag[j] == 1):  # hot extreme
                    is_hot = True
                    break

            # if hot extreme found
            if ((is_cold == True) & (is_hot == True)):
                cold2hot_loc.append([i, j])
                i = j+1  # search the next event
        else:
            is_cold = False
            
    cold2hot_loc = np.asarray(cold2hot_loc) ## convert from list to array
    if (len(cold2hot_loc)<=0):
        return []

    ## remove the repeated event (i.e., having the same index for hot extreme)
    repeated_idx = np.where((cold2hot_loc[0:len(cold2hot_loc)-1, 1] - cold2hot_loc[1:len(cold2hot_loc), 1])==0)[0]
    cold2hot_loc = np.delete(cold2hot_loc, repeated_idx, 0)
    
    col_nan = np.full((len(cold2hot_loc), 1), np.nan)
    cold2hot_loc = np.c_[cold2hot_loc, col_nan, col_nan]
    
    for row in range(len(cold2hot_loc)):
        t_start = int(cold2hot_loc[row][0])
        t_end = int(cold2hot_loc[row][1])
        
        ## 获取cold-to-warm发生之前，出现-1的最后一天
        t_start_cold = np.array(np.where(flag[:t_start+1]>-1))[0][-1]+1
        
        ## 获取cold-to-warm之后，结束1的最后一天
        t_end_warm = t_end+np.array(np.where(flag[t_end:]<1))[0][0]-1
        
        cold2hot_loc[row][2] =  t_start_cold 
        cold2hot_loc[row][3] =  t_end_warm 
        
    return cold2hot_loc

def warm2cool_whiplash(flag, max_gap=5):
    """ Identify hot2cold temperature whiplash events
    
    Parameters
    ----------
    flag : temperature flag series (1: hot, -1: cold, 0: non-hot & non-cold)
    max_gap : the maximum gap between the start and end of the event
        
    
    Returns
    -------
    hot2cold_loc: two-column arrary with the first/second column marking the start/end of the whiplash event
    """
    
    ## length of the time series
    n = len(flag)
    
    ## index location of the identified hot2cold events
    hot2cold_loc = [] 
    
    ## search for hot2cold  events
    for i in range(0, n):
        if (flag[i] == 1):  # if hot extreme found
            is_cold = False
            is_hot = True

            # search for cold extreme in the following 'max_gap' days
            for j in range(i+1, min(i+max_gap+1, n)):
                if (flag[j] == -1):  # cold extreme
                    is_cold = True
                    break

            # if hot extreme found
            if ((is_hot == True) & (is_cold == True)):
                hot2cold_loc.append([i, j])
                i = j+1  # search the next event
        else:
            is_hot = False
            
    hot2cold_loc = np.asarray(hot2cold_loc) ## convert from list to array

    if (len(hot2cold_loc)<=0):
        return []

    ## remove the repeated event (i.e., having the same index for hot extreme)
    repeated_idx = np.where((hot2cold_loc[0:len(hot2cold_loc)-1, 1] - hot2cold_loc[1:len(hot2cold_loc), 1]) ==0)[0]
    hot2cold_loc = np.delete(hot2cold_loc, repeated_idx, 0)
    
    col_nan = np.full((len(hot2cold_loc), 1), np.nan)
    hot2cold_loc = np.c_[hot2cold_loc, col_nan, col_nan]
    
    for row in range (len(hot2cold_loc)):
        t_start = int(hot2cold_loc[row][0])
        t_end = int(hot2cold_loc[row][1])
        
        ## 获取warm-to-cold发生之前，出现1的最后一天
        t_start_warm = np.array(np.where(flag[:t_start+1]<1))[0][-1]+1

        ## 获取warm-to-cold之后，结束1的最后一天
        t_end_cold = t_end+np.array(np.where(flag[t_end:]>-1))[0][0]-1
        
        hot2cold_loc[row][2] =  t_start_warm 
        hot2cold_loc[row][3] =  t_end_cold
    
    return hot2cold_loc
        

def land_sea_mask(data, time = True, time_var = 'time', no_Antarctica = False):
    ## input xarray
    land_sea_mask = xr.open_dataset(r'Z:/Public/5_Others/3D_sjwu/01sjwu/land_sea_mask.2.5x2.5.nc')   # ERA5 mask文件
    # land_sea_mask = land_sea_mask.rename({'longitude':'lon', 'latitude':'lat'})
    if time == True:
        land_sea_mask_= np.tile(land_sea_mask['lsm'],(len(data[time_var]),1,1))
    else:
        land_sea_mask_ = land_sea_mask['lsm'][0,:,:].values
        
    land_sea_mask_[land_sea_mask_<=0] = np.nan
    land_sea_mask_[land_sea_mask_>0] = 1
    data = data* land_sea_mask_
    
    if no_Antarctica == True:
        # 将纬度在-60到90之间的南极洲区域设置为NaN
        mask = (data['lat'] < -60)# 创建南极洲的掩膜
        # 将南极洲区域设置为NaN
        data = data.where(~mask)
        
    return data




def detrend_with_rolling(data, window=31, order=3):
    
    lat_len = len(data['lat'])
    lon_len = len(data['lon'])       
    ## Assign memory to save the results
    data_detrended = data.copy(deep=True)
    data_detrended[:] = np.nan
    
    ## Perform the detrending process
    plusminus = window // 2
    for doy in tqdm(range(1, 367)):
        print(doy)
        
        # For a given 'day', obtain a window around that day.
        center_index = data.time.dt.dayofyear.isin(doy)
        window_doys = (np.arange(doy - plusminus - 1, doy + plusminus) % 366) + 1
        window_index = data.time.dt.dayofyear.isin(window_doys)
       
        # Get the data in the window and average it by time.year
        temp = data[window_index, :].groupby("time.year").mean()
    
        # Detrend by polynomial fitting
        x = temp["year"]
        for iy in range(lat_len):
            for ix in range(lon_len):
                # if mask[iy, ix] == False:
                #     continue]
                y = temp[:, iy, ix].values
                idx = np.isfinite(y)
                if sum(idx)< 0.8 * len(y):
                    continue 
                
                trend = np.polyval(np.polyfit(x[idx].values, y[idx], deg=order), x)
                idx_year = (data[center_index, iy, ix]["time.year"] - temp["year"].min()).values
                data_detrended[center_index, iy, ix] = data[center_index, iy, ix] - trend[idx_year]   

    return data_detrended


def detrend_polyfit(x, y, order=2): 
    """ Remove the linear trend of a time series
    用2次多项式拟合，并去趋势
    Parameters
    ----------
    x : Independent variable (e.g., year)
    y : Dependent variable

    Returns
    -------
    detrend_y: detrended data
    """
    
    idx = np.isfinite(y)
    if sum(idx)< 0.8 * len(y):
        return np.full(y.size, np.nan)
    
    trend = np.polyval(np.polyfit(x[idx], y[idx], deg=order), x)
    return y - trend


#%% [0] Input path and parameter
'''
Input data is daily data
Then we rolling detrend this data
对原始的daily数据进行rolling mean是为了使得数据平滑，去除一些偶然出现的高值的影响。

'''
dataset_name = 'HadGHCND_24'
region_list = ['Global','Arid', 'Temperate','Tropical','Polar', 'Cold']
region_color = {'Arid': '#F2D599',
                'Temperate': '#D68582',
                'Tropical': '#B7ADE9',
                'Polar':'#79D9EF',
                'Cold':'#44A4A0'}
season_list = ["Annual", "DJF", "MAM", "JJA", "SON"]

dataset_name_list = ["ERA5","Berkeley","NCEP1"]  ##
# dataset_name_list = ["HadGHCND_24"] 
# dataset_name_list = ["NOAA_DOE"] 

whiplash = ["W2C", "C2W"]
whiplash_names = {"W2C": "Hot-to-Cold","C2W": "Cold-to-Hot"}

whiplash_keys = ["Frequency", "Maximum Intensity", "Transition Duration"]; 
n_keys = len(whiplash_keys)

relative_to_baseline=False
if relative_to_baseline==True:
    whiplash_keys_units = {"Frequency": "(%)", 
                           "Maximum Intensity": "(%)",
                           "Transition Duration": "(%)"}
    whiplash_keys_trend_units = {"Frequency": "(%/100yr)", 
                                 "Maximum Intensity": "(%/100yr)",
                                 "Transition Duration": "(%/100yr)"}
else:
    whiplash_keys_units = {"Frequency": "(events)", 
                           "Maximum Intensity": "(s.d.)",
                           "Transition Duration": "(days)"}
    whiplash_keys_trend_units = {"Frequency": "(events/100yr)", 
                                 "Maximum Intensity": "(s.d./100yr)",
                                 "Transition Duration": "(days/100yr)"}

dataset_startyear = {"ERA5": 1940,
                     "Berkeley": 1950,
                     "HadGHCND": 1950,
                     "HadGHCND_24": 1950,
                     "MERRA2": 1980,
                     "NCEP2": 1979,
                     "NCEP1": 1948,
                     "CPC": 1979,
                     "JRA55": 1958,
                     "NOAA_DOE":1900} 

dataset_endyear = {"ERA5": 2023,
                   "Berkeley": 2021,
                   "HadGHCND": 2014,
                   "HadGHCND_24": 2024,
                   "MERRA2": 2022,
                   "NCEP2": 2022,
                   "NCEP1": 2023,
                   "CPC": 2022,
                   "JRA55": 2023,
                   "NOAA_DOE":2015} 

dataset_file_names = {"ERA5": "ERA5.global.2m_temperature.daily.2.5x2.5.1950_2022.nc",
                      "Berkeley": "Berkeley.global.tavg.19500101-20211231.2.5x2.5.nc",
                      "HadGHCND": "HadGHCND.global.tmean.daily.1950_2014.nc",
                      "HadGHCND_24": "daily.tmean.anom.ghcnd.grid.1950-2024.nc",
                      "MERRA2": "MERRA2.global.2m_temperature.daily.2.5x2.5.1980_2022.nc",
                      "NCEP2": "NCEP2.global.2m_temperature.daily.2.5x2.5.1979_2022.nc",
                      "NCEP1": "NCEP1.global.2m_temperature.daily.2.5x2.5.1948_2023.nc",
                      "CPC": "cpc.global.tmean.daily.2.5x2.5.1979_2022.nc",
                      "JRA55": "JRA55.global.2m_temperature.daily.2.5x2.5.1958_2023.nc",
                      "NOAA_DOE":"NOAA_CIRES-DOE_20th_Century_Reanalysis_V3_air.2m.1836-2015.regrid_2.5x2.5.nc"}

if dataset_name == 'ERA5':
    ref_period = slice('1961-01-01', '1990-12-31')
elif dataset_name == 'NOAA_DOE':
    ref_period = slice('1900-01-01', '1929-12-31')
elif dataset_name == 'HadGHCND_24':
    ref_period = slice('1961-01-01', '1990-12-31')
else:
    ref_period = slice('1961-01-01', '1990-12-31')

cycle_window = 31
max_gap = 5
roll_window =max_gap
detrend_order = 3
std = 1
method = 'Method_2'
IPCC = 'AR6'

if  method == 'Method_2':
    '''
    method 2: 
        (1) 首先对输入的原始气温数据tas去趋势，获得Detrended tas
        (2) Detrended tas进行5d滑动平均
        (3）计算std, 获取Standardized tas
    '''
    print('method 2')
    _var = 'tas'

    
'''
(1) daily tas——daily anomaly
(2) daily anomaly——detrend
(3) detrended anoma——5d滑动平均
(4) detrended 5d anom——算std——standardized
注：(2)和(3)可以交换顺序尝试
整体detrended
'''

detrended = 'detrended_rolling'

work_dir = "G:/01PhD/01-25 Temperature_whiplash_no_bias"+"//"+ method
work_dir_out = "G:/01PhD/01-25 Temperature_whiplash_no_bias"+"//"+ method


#%%[1] Process raw data

#%%% [1-1] Detrended data
'''
Detrend raw data to obtain detrended tas

'''

ds = xr.open_dataset(work_dir+ '/01_Data'+'//' + dataset_file_names[dataset_name])


if dataset_name == "HadGHCND":
    ds_ERA5 = xr.open_dataset(work_dir  +'/01_Data'+'//' + dataset_file_names["ERA5"])
    ds = ds.interp(lat=ds_ERA5.lat.values, lon=ds_ERA5.lon.values)   #插值 
elif dataset_name == "NCEP1":
    ds = ds.drop_vars(["time_bnds"])
elif dataset_name == "JRA55":
    ds = ds.drop_vars(["initial_time0_hours_bnds", "initial_time0_encoded"])
    ds = ds.rename({'initial_time0_hours':'time'}) 
elif dataset_name == "NOAA_DOE":
    ds = ds.drop_vars(["time_bnds"])
    ds = ds.sel(time=slice('1850-01-01', '2015-12-31'))
    
for var in list(ds.keys()):
    print(var)
ds = ds[var].to_dataset(name = 'tas')

## 掩膜陆地
if dataset_name == "HadGHCND_24":
    ds['tas'] = region_mask(ds['tas'])
else:
    ds['tas'] = land_sea_mask(ds['tas'], time = True, time_var = 'time', no_Antarctica = True)

## Detrend data
if  method == 'Method_2':
    print('method 2')
    ## Detrend standardized tas
    if detrended == 'detrended_rolling':
        print('detrended_rolling')
        ## 对所有年每天前后15天的进行去趋势
        ds['Detrended '+_var] = detrend_with_rolling(ds[_var], window=cycle_window, order=detrend_order)
    else:
        print('detrended')
        ds['Detrended '+_var] = copy.deepcopy(ds[_var])
        ds['Detrended '+_var].values[:] = np.nan
        x = np.arange(0, len(ds['time']))
        for iy in range(ds['lat'].size):
            for ix in range(ds['lon'].size):
                # print(iy,ix)
                ds['Detrended '+_var][:, iy, ix] = detrend_polyfit(x, ds[_var][:, iy, ix].values, order = detrend_order)
        del ix, iy, x

## Save detrended data
ds.to_netcdf('Z:/Public/1_Datasets/1_ClimateData/NOAA_CIRES-DOE_20th_Century_Reanalysis_V3/'+dataset_name + '_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'.nc')

# ds.to_netcdf('Z:/Public/1_Datasets/1_ClimateData/HadGHCND/'+dataset_name + '_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'.nc')


# ds.to_netcdf(work_dir + '/01_Data/01_Other_'+detrended+'//'+dataset_name + '_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'.nc')
del ds
gc.collect()


#%%% [1-2] Input data and Detect cold-hot / hot-cold

## Added detrended data
# ds = xr.open_dataset('Z:/Public/1_Datasets/1_ClimateData/NOAA_CIRES-DOE_20th_Century_Reanalysis_V3/'+dataset_name +'_detrended_rolling_tas_polyfit_order-3'+'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'.nc')
ds = xr.open_dataset(work_dir + '/01_Data/01_Other_'+detrended+'//'+dataset_name + '_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'.nc')
# ds = xr.open_dataset('Z:/Public/1_Datasets/1_ClimateData/HadGHCND/'+dataset_name + '_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'.nc')


_var_intensity = 'Standardized '+_var

if (method == 'Method_1') | (method == 'Method_4'):
    print('method 1 or 4')
    print(_var)
    ## calculate std of 'Detrended anom'
    clim_std = moving_std(ds['Detrended '+_var].sel(time = ref_period), window = cycle_window)  # 31-day sliding std
    
    ## standardized anomalies
    ds['Standardized '+_var]  = ds['Detrended '+_var].groupby('time.dayofyear')/clim_std   ## standardized anomalies

elif (method == 'Method_2') | (method == 'Method_3'):
    print('method 2 or 3')
    print(_var)
    ## rolling mean 5d
    ds['Detrended '+_var] = ds['Detrended '+_var].rolling(center=True, time = roll_window).mean()

    ## calculate std
    clim_std = moving_std(ds['Detrended '+_var].sel(time = ref_period),  window = cycle_window)  # 15-day sliding std

    ## standardized anomalies
    ds['Standardized '+_var]  = ds['Detrended '+_var].groupby('time.dayofyear')/clim_std   ## standardized anomalies
    del clim_std
    gc.collect()

## Obtain -1(cool) or 1(warm) time series
"""
warm： 大于1倍标准差
cool： 小于1倍标准差


"""
## 获取-1, 1的序列（冷天、热天）
ds['Extreme'] = (ds[_var_intensity]>std) + 0   ## hot extreme：大于1倍标准差 记为1
ds['Extreme'].values[(ds[_var_intensity]<-std)] = -1 ## cold extreme：小于1倍标准差 记为-1

# ds['Extreme']  = ds['Extreme'] * (ds['tas']/ds['tas'])

ds['Extreme'] = land_sea_mask(ds['Extreme'], time = True, time_var = 'time', no_Antarctica = True)

## identify temperature whiplash
ds_event = ds[_var_intensity].copy(deep=True).to_dataset(name='C2W')
ds_event['C2W'].values[:] = np.nan
ds_event['C2W Transition Intensity'] = ds_event['C2W'].copy(deep=True);
ds_event['C2W Maximum Intensity'] = ds_event['C2W'].copy(deep=True);
ds_event['C2W Transition Duration']  = ds_event['C2W'].copy(deep=True);
ds_event['C2W Maximum Duration']  = ds_event['C2W'].copy(deep=True);

ds_event['W2C']           = ds_event['C2W'].copy(deep=True);
ds_event['W2C Transition Intensity'] = ds_event['C2W'].copy(deep=True)
ds_event['W2C Maximum Intensity'] = ds_event['C2W'].copy(deep=True)
ds_event['W2C Transition Duration']  = ds_event['C2W'].copy(deep=True);
ds_event['W2C Maximum Duration']  = ds_event['C2W'].copy(deep=True);

for iy in tqdm(range(len(ds['lat']))):
    for ix in range(len(ds['lon'])):
        flag = ds['Extreme'].values[:, iy, ix].flatten()  ## 格网[iy, ix]用于判定whiplash事件的序列
        # print(iy,ix, np.sum(flag))
        ## cold-to-hot whiplash events
        c2h_loc = cool2warm_whiplash(flag, max_gap=max_gap)
        if (len(c2h_loc)>0):
            c2h_loc = c2h_loc.astype(int)
            
            ds_event['C2W'][c2h_loc[:, 0], iy, ix]           = 1  ## occur cold extreme
            ds_event['C2W'][c2h_loc[:, 1], iy, ix]           = 0 ## occur hot extreme
            
            ds_event['C2W Transition Duration'][c2h_loc[:, 1], iy, ix]  = (c2h_loc[:, 1] - c2h_loc[:, 0])
            ds_event['C2W Maximum Duration'][c2h_loc[:, 1], iy, ix]  = (c2h_loc[:, 3] - c2h_loc[:, 2])
            ds_event['C2W Transition Intensity'][c2h_loc[:, 1], iy, ix] = abs(ds[_var_intensity].values[c2h_loc[:, 1], iy, ix] - ds[_var_intensity].values[c2h_loc[:, 0], iy, ix])
            ds_event['C2W Maximum Intensity'][c2h_loc[:, 1], iy, ix]= np.array([np.max(ds[_var_intensity].values[c2h_loc[i, 1]:c2h_loc[i,3]+1, iy, ix]) for i in np.arange(0, c2h_loc.shape[0])])\
                - np.array([np.min(ds[_var_intensity].values[c2h_loc[i, 2]:c2h_loc[i,0]+1, iy, ix]) for i in np.arange(0, c2h_loc.shape[0])])
        
        ## cold-to-hot whiplash events
        h2c_loc = warm2cool_whiplash(flag, max_gap=max_gap)
        if (len(h2c_loc)>0):
            h2c_loc =h2c_loc.astype(int)
            
            ds_event['W2C'][h2c_loc[:, 0], iy, ix]           = 1   ## occur hot extreme
            ds_event['W2C'][h2c_loc[:, 1], iy, ix]           = 0  ## occur cold extreme 
            ds_event['W2C Transition Duration'][h2c_loc[:, 1], iy, ix]  = (h2c_loc[:, 1] - h2c_loc[:, 0])
            ds_event['W2C Maximum Duration'][h2c_loc[:, 1], iy, ix]  = (h2c_loc[:, 3] - h2c_loc[:, 2])
            ds_event['W2C Transition Intensity'][h2c_loc[:, 1], iy, ix] = abs(ds[_var_intensity].values[h2c_loc[:, 1], iy, ix] - ds[_var_intensity].values[h2c_loc[:, 0], iy, ix])
            ds_event['W2C Maximum Intensity'][h2c_loc[:, 1], iy, ix] = np.array([np.max(ds[_var_intensity].values[h2c_loc[i, 2]:h2c_loc[i,0]+1, iy, ix]) for i in np.arange(0, h2c_loc.shape[0])])\
                - np.array([np.min(ds[_var_intensity].values[h2c_loc[i, 1]:h2c_loc[i,3]+1, iy, ix]) for i in np.arange(0, h2c_loc.shape[0])])
        
        
del iy, ix, flag, c2h_loc, h2c_loc
gc.collect()

out_file_name = dataset_name+'_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) +'_'+str(ds['time.year'][0].values) + '-' +str(ds['time.year'][-1].values) +'ref_1900-1929.nc'
del ds
gc.collect()


ds_event.to_netcdf(path = work_dir + '/02_Result/01_Other_result_'+detrended+'//'+'01_event'+'//'+'whiplash_event-'+str(max_gap) +'//'+'std-'+str(std)+'//'+out_file_name)
del ds_event, out_file_name
gc.collect()


#%% [3] Calculate seasonal series
#%%% [3-1] Calculate seasonal mean
'''
"Annual", "DJF", "MAM", "JJA", "SON"

'''
file_name = dataset_name+'_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+ 'ref_1900-1929.nc'
ds_event = xr.open_dataset(work_dir + '/02_Result/01_Other_result_'+detrended+'/01_event/whiplash_event-'+str(max_gap)+'//'+'std-'+str(std)+'//'+ file_name)  
# ds_event= xr.open_dataset('Z:/Public/1_Datasets/1_ClimateData/NOAA_CIRES-DOE_20th_Century_Reanalysis_V3' +'//'+file_name)
## calculate seasonal mean
ds_season_mean = season_mean(ds_event, concat=True)
ds_event = ds_event.drop(['C2W Transition Intensity', 'C2W Maximum Intensity',
                          'C2W Transition Duration',  'C2W Maximum Duration', 
                          'W2C Transition Intensity', 'W2C Maximum Intensity',
                          'W2C Transition Duration',  'W2C Maximum Duration',])  ## 保留用于计算Frequency的变量即可
ds_season_sum = season_sum(ds_event, concat=True)  ## Frequency
del ds_event
gc.collect()

ds_season_mean = ds_season_mean.rename({"C2W":"C2W Frequency", "W2C":"W2C Frequency"})
ds_season_mean['C2W Frequency'] = ds_season_sum['C2W']
ds_season_mean['W2C Frequency'] = ds_season_sum['W2C']
del ds_season_sum
gc.collect()

out_file_name='seasonal_'+dataset_name+'_'+detrended+'_'+_var+'_polyfit_order-'+str(detrend_order)+'_'+'whiplash_event-'+str(max_gap) +'_'+str(dataset_startyear[dataset_name]) +'-'+ str(dataset_endyear[dataset_name])+'ref_1900-1929.nc'
ds_season_mean.to_netcdf(work_dir +'/02_Result/01_Other_result_'+detrended+'/02_seasonal/whiplash_event-'+str(max_gap) +'//'+'std-'+str(std)+'//'+ out_file_name)  
# ds_season_mean.to_netcdf('Z:/Public/1_Datasets/1_ClimateData/NOAA_CIRES-DOE_20th_Century_Reanalysis_V3'+'//'+out_file_name)  

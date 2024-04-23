# %%
# librairies

import xarray as xr
import numpy as np
import pandas as pd
import sys
import gc

# %%
# variable that I want to extract for each storm

folder = (str(sys.argv[1]))

year = (str(sys.argv[2]))

print(folder)
print(year)

year = int(year)
year_next = year + 1

year = str(year)
year_next = str(year_next)

month_act = [10 ,11, 12]
month_next = [1, 2, 3]

#var = 'ERA5_' + str(year) + '_' + folder

# %%
# new version for combining 6 months from one year to the other

way = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/SL/'

if year == '1990' or year == '2021':
    if year == '1990':
        month_act = [1, 2, 3, 10, 11, 12]
        for i in month_act:
            var = 'ERA5_' + year + '-' + str(i) + '_' + folder
            locals()['dew_point_xr_' + str(i)] = xr.open_dataset(way+folder+'/'+var+'.nc')

        dew_point_xr_1 = xr.concat([dew_point_xr_1, dew_point_xr_2, dew_point_xr_3, dew_point_xr_10, dew_point_xr_11, dew_point_xr_12], dim='time')
        dew_point_xr_to_combined = dew_point_xr_1
        del dew_point_xr_1, dew_point_xr_2, dew_point_xr_3, dew_point_xr_10, dew_point_xr_11, dew_point_xr_12
        gc.collect()
        dew_point_xr = dew_point_xr_to_combined 
        del dew_point_xr_to_combined

    elif year == '2021':
        month_next = [1, 2, 3, 10, 11, 12]
        for i in month_next:
            next_var = 'ERA5_' + year +'-' + str(i) + '_' + folder
            locals()['dew_point_xr_' + str(i)] = xr.open_dataset(way+folder+'/'+next_var+'.nc')

        dew_point_xr_10 = xr.concat([dew_point_xr_1, dew_point_xr_2, dew_point_xr_3, dew_point_xr_10, dew_point_xr_11, dew_point_xr_12], dim='time')
        dew_point_xr_to_combined = dew_point_xr_10
        del dew_point_xr_1, dew_point_xr_2, dew_point_xr_3, dew_point_xr_10, dew_point_xr_11, dew_point_xr_12
        gc.collect()
        dew_point_xr = dew_point_xr_to_combined 
        del dew_point_xr_to_combined

else:
    for i in month_act:
            var = 'ERA5_' + year + '-' + str(i) + '_' + folder
            locals()['dew_point_xr_' + str(i)] = xr.open_dataset(way+folder+'/'+var+'.nc')
            
    for i in month_next:
        next_var = 'ERA5_' + year_next +'-' + str(i) + '_' + folder
        locals()['dew_point_xr_' + str(i)] = xr.open_dataset(way+folder+'/'+next_var+'.nc')

    dew_point_xr_10 = xr.concat([dew_point_xr_10, dew_point_xr_11, dew_point_xr_12,dew_point_xr_1, dew_point_xr_2, dew_point_xr_3], dim='time')
    dew_point_xr_to_combined = dew_point_xr_10
    del dew_point_xr_1, dew_point_xr_2, dew_point_xr_3, dew_point_xr_10, dew_point_xr_11, dew_point_xr_12
    gc.collect()
    dew_point_xr = dew_point_xr_to_combined 
    del dew_point_xr_to_combined


specific_var = list(dew_point_xr.variables)[0]

if specific_var == 'longitude':
    specific_var = list(dew_point_xr.variables)[3]

# %%
# import all tracks

dates = pd.read_csv('/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/storms_start_end.csv', sep=',')
dates['year'] = dates['start_date'].str[:4]

# Convert 'start_date' and 'end_date' columns to datetime objects
dates['start_date'] = pd.to_datetime(dates['start_date'])
dates['end_date'] = pd.to_datetime(dates['end_date'])

# Find the index of the first storm that begins on or after October
index_start_october = None
for index, row in dates.iterrows():
    if row['start_date'].month >= 10 and row['start_date'].year == int(year):
        index_start_october = index
        break

# Find the index of the last storm that ends on or before March
index_end_march = None
for index, row in dates.iterrows():
    if row['end_date'].month <= 3 and row['end_date'].year == int(year_next):
        index_end_march = index

# Count the number of storms occurring between October and March within the specified timeframe
#num_storms_october_to_march = index_end_march - index_start_october + 1

'''print("Index of the first storm that begins on or after October within the specified timeframe:", index_start_october)
print("Index of the last storm that ends on or before March within the specified timeframe:", index_end_march)
print("Number of storms occurring between October and March within the specified timeframe:", num_storms_october_to_march)'''


for i in range(index_start_october,index_end_march):
    locals()['track_' + str(i+1)] = pd.read_csv('/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/tc_irad_tracks/tc_irad_1_hour/tc_irad_' + str(i+1) + '_interp.txt')   

# %%
# slice the time dimension

for i in range(index_start_october,index_end_march+1):
    start_temp = dates['start_date'][i]
    end_temp = dates['end_date'][i]

    # Create a new dataset for each iteration
    new_dataset = xr.Dataset({
        specific_var: dew_point_xr[specific_var],
        # Add other variables as needed
    })

    # Optionally, you can update the time dimension for the new dataset
    new_dataset = new_dataset.sel(time=slice(start_temp, end_temp))

    # Dynamically name the dataset variable
    locals()[f"dew_point_xr_{i}"] = new_dataset

# %%
# iterate through each time step

var_out = []
var_out
for j in range (index_start_october,index_end_march+1):
    track_temp = pd.read_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/tc_irad_tracks/tc_1_hour/tc_irad_{j+1}_interp.txt')
    dew_point_temp = locals()[f"dew_point_xr_{j}"]
    var_out_temp = []
    for i in range(0, len(track_temp)):
        lon_e_temp, lon_w_temp, lat_s_temp, lat_n_temp = track_temp.iloc[i]

        lon_test = np.asanyarray(dew_point_temp.longitude[:])
        lat_test = np.asanyarray(dew_point_temp.latitude[:])

        closest_lon_w = np.abs(lon_test - lon_w_temp).argmin()
        closest_lon_e = np.abs(lon_test - lon_e_temp).argmin()
        closest_lat_s = np.abs(lat_test - lat_s_temp).argmin()
        closest_lat_n = np.abs(lat_test - lat_n_temp).argmin()

        closest_lon_w_coor = lon_test[closest_lon_w]
        closest_lon_e_coor = lon_test[closest_lon_e]
        closest_lat_s_coor = lat_test[closest_lat_s]
        closest_lat_n_coor = lat_test[closest_lat_n]

        # Use .roll to handle the 0°/360° boundary

        if closest_lon_w_coor < 100 and closest_lon_e_coor > 100:
            roll_shift = {'longitude':int(round(closest_lon_w_coor,0)), 'longitude': int(round(closest_lon_e_coor,0))}
            dew_point_temp_rolled = dew_point_temp.roll(roll_shift, roll_coords=True)
        else:
            dew_point_temp_rolled = dew_point_temp
        # Slice the dataset based on the rolled longitudes and latitudes
        temp_ds_time = dew_point_temp_rolled[specific_var].isel(time=i)
        temp_ds = temp_ds_time.sel(latitude=slice(closest_lat_n_coor, closest_lat_s_coor),
                                   longitude=slice(closest_lon_e_coor, closest_lon_w_coor)).values

        #temp_ds = np.asarray(dew_point_temp['d2m'])

        #var_out_temp.append(temp_ds[i,closest_lat_n:closest_lat_s,closest_lon_e:closest_lon_w])
        var_out_temp.append(temp_ds)

    locals()[f"var_out_{j+1}"] = var_out_temp

# %%
# mean, min, max value of each time step

mean_out = []
min_out = []
max_out = []
sigma_out = []
skweness_out = []
kurto_out = []

stats_list = []
all_stats = []
for j in range(index_start_october,index_end_march+1):

    var_out = locals()[f"var_out_{j+1}"]
    mean_out_temp = []
    min_out_temp = []
    max_out_temp = []
    sigma_out_temp = []
    skweness_out_temp = []
    kurto_out_temp = []

    for i in range(0, len(var_out)):
        mean_out_temp.append(np.mean(var_out[i]))
        min_out_temp.append(np.min(var_out[i]))
        max_out_temp.append(np.max(var_out[i]))
        sigma_out_temp.append(np.std(var_out[i]))
        skweness_out_temp.append(pd.Series(np.asarray(var_out[i]).reshape(-1)).skew())
        kurto_out_temp.append(pd.Series(np.asarray(var_out[i]).reshape(-1)).kurtosis())
    
    mean_out.append(mean_out_temp)
    min_out.append(min_out_temp)
    max_out.append(max_out_temp)
    sigma_out.append(sigma_out_temp)
    skweness_out.append(skweness_out_temp)
    kurto_out.append(kurto_out_temp)

    locals()[f"mean_out_{j+1}"] = mean_out_temp
    locals()[f"min_out_{j+1}"] = min_out_temp
    locals()[f"max_out_{j+1}"] = max_out_temp
    locals()[f"sigma_out_{j+1}"] = sigma_out_temp
    locals()[f"skweness_out_{j+1}"] = skweness_out_temp
    locals()[f"kurto_out_{j+1}"] = kurto_out_temp

# %%
# save as csv

for j in range(index_start_october,index_end_march+1):

    locals()[f"mean_out_{j+1}"] = pd.DataFrame(locals()[f"mean_out_{j+1}"])
    locals()[f"min_out_{j+1}"] = pd.DataFrame(locals()[f"min_out_{j+1}"])
    locals()[f"max_out_{j+1}"] = pd.DataFrame(locals()[f"max_out_{j+1}"])
    locals()[f"sigma_out_{j+1}"] = pd.DataFrame(locals()[f"sigma_out_{j+1}"])
    locals()[f"skweness_out_{j+1}"] = pd.DataFrame(locals()[f"skweness_out_{j+1}"])
    locals()[f"kurto_out_{j+1}"] = pd.DataFrame(locals()[f"kurto_out_{j+1}"])

    locals()[f"mean_out_{j+1}"].to_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/datasets/{folder}/storm_{j+1}/mean_{j+1}.csv')
    locals()[f"min_out_{j+1}"].to_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/datasets/{folder}/storm_{j+1}/min_{j+1}.csv')
    locals()[f"max_out_{j+1}"].to_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/datasets/{folder}/storm_{j+1}/max_{j+1}.csv')
    locals()[f"sigma_out_{j+1}"].to_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/datasets/{folder}/storm_{j+1}/sigma_{j+1}.csv')
    #locals()[f"skweness_out_{j+1}"].to_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/datasets/{folder}/storm_{j+1}/skweness_{j+1}.csv')
    #locals()[f"kurto_out_{j+1}"].to_csv(f'/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/datasets/{folder}/storm_{j+1}/kurto_{j+1}.csv')

# %% [markdown]
# For all variables, (class by variable (folder), then by storm)) 
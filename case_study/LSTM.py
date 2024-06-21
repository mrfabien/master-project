# %%
# creation of a simple regression model with the mean of each variable in datasets_3h folder
# and the mean of the target variable in the training set

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten
from tensorflow.keras import mixed_precision
import shap
from sklearn.inspection import permutation_importance
import pickle

op ='win'
if op == 'win':
    path = f'~/OneDrive/Documents/GitHub/master-project/'
else:
    path = f'~/Documents/GitHub/master-project/'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
#one_storm = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/instantaneous_10m_wind_gust/storm_1/max_1_0.csv')

# select storm
storms = range(1,97)
dataset = 'datasets_1h'

for storm in storms:
    try:
        one_storm_pd = pd.read_csv(path+f'DATASETS/{dataset}/instantaneous_10m_wind_gust/storm_{storm}/max_{storm}_0.csv')
        one_storm_EU_pd = pd.read_csv(path+f'DATASETS/{dataset}_EU/instantaneous_10m_wind_gust/storm_{storm}/max_{storm}_0.csv')
        one_storm_mean = pd.read_csv(path+f'DATASETS/{dataset}/instantaneous_10m_wind_gust/storm_{storm}/mean_{storm}_0.csv')
        one_storm_EU_mean = pd.read_csv(path+f'DATASETS/{dataset}_EU/instantaneous_10m_wind_gust/storm_{storm}/mean_{storm}_0.csv')
        one_storm_min = pd.read_csv(path+f'DATASETS/{dataset}/instantaneous_10m_wind_gust/storm_{storm}/min_{storm}_0.csv')
        one_storm_EU_min = pd.read_csv(path+f'DATASETS/{dataset}_EU/instantaneous_10m_wind_gust/storm_{storm}/min_{storm}_0.csv')
        one_storm_std = pd.read_csv(path+f'DATASETS/{dataset}/instantaneous_10m_wind_gust/storm_{storm}/std_{storm}_0.csv')
        one_storm_EU_std = pd.read_csv(path+f'DATASETS/{dataset}_EU/instantaneous_10m_wind_gust/storm_{storm}/std_{storm}_0.csv')

    except FileNotFoundError:
        print(f"Fichier pour la tempête {storm} non trouvé.")
        continue

    one_storm = np.asarray(one_storm_pd['0'])
    one_storm_EU = np.asarray(one_storm_EU_pd['0'])
    plt.plot(one_storm_pd['Unnamed: 0'], one_storm,label='Max of the whole path')#, lw=0, marker='o')
    plt.scatter(one_storm_EU_pd['Unnamed: 0'],one_storm_EU, label='Max of the path in EU borders')#, lw=0, marker='o')
    plt.plot(one_storm_mean['Unnamed: 0'], one_storm_mean['0'], label='Mean of the whole path')
    plt.scatter(one_storm_EU_mean['Unnamed: 0'], one_storm_EU_mean['0'], label='Mean of the path in EU borders')
    plt.plot(one_storm_min['Unnamed: 0'], one_storm_min['0'], label='Min of the whole path')
    plt.scatter(one_storm_EU_min['Unnamed: 0'], one_storm_EU_min['0'], label='Min of the path in EU borders')
    plt.plot(one_storm_std['Unnamed: 0'], one_storm_std['0'], label='Std of the whole path')
    plt.scatter(one_storm_EU_std['Unnamed: 0'], one_storm_EU_std['0'], label='Std of the path in EU borders')


    plt.title(f'Comparison of the different datasets for wind gust in storm {storm}')
    plt.xlabel('Step')
    plt.ylabel('Wind Gust in m/s')
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# %%
# do the same for the other variables such as relative humidity, temperature, etc.

storms = range(25,26)
dataset = 'datasets_1h'
variable = 'vertical_velocity'
full_pressure = [
    "0",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "850",
    "900",
    "925",
    "950",
    "975",
    "1000",
]

for storm in storms:
    for level in full_pressure:
        try:
            one_storm_pd = pd.read_csv(path+f'DATASETS/{dataset}/{variable}/storm_{storm}/max_{storm}_{level}.csv')
            one_storm_EU_pd = pd.read_csv(path+f'DATASETS/{dataset}_EU/{variable}/storm_{storm}/max_{storm}_{level}.csv')
            one_storm_mean = pd.read_csv(path+f'DATASETS/{dataset}/{variable}/storm_{storm}/mean_{storm}_{level}.csv')
            one_storm_EU_mean = pd.read_csv(path+f'DATASETS/{dataset}_EU/{variable}/storm_{storm}/mean_{storm}_{level}.csv')
            one_storm_min = pd.read_csv(path+f'DATASETS/{dataset}/{variable}/storm_{storm}/min_{storm}_{level}.csv')
            one_storm_EU_min = pd.read_csv(path+f'DATASETS/{dataset}_EU/{variable}/storm_{storm}/min_{storm}_{level}.csv')
            one_storm_std = pd.read_csv(path+f'DATASETS/{dataset}/{variable}/storm_{storm}/std_{storm}_{level}.csv')
            one_storm_EU_std = pd.read_csv(path+f'DATASETS/{dataset}_EU/{variable}/storm_{storm}/std_{storm}_{level}.csv')

        except FileNotFoundError:
            print(f"Fichier pour la tempête {storm} non trouvé.")
            continue

        one_storm = np.asarray(one_storm_pd['0'])
        #one_storm_EU = np.asarray(one_storm_EU_pd['0'])
        #plt.plot(one_storm_pd['Unnamed: 0'], one_storm,label='Max of the whole path')#, lw=0, marker='o')
        #plt.scatter(one_storm_EU_pd['Unnamed: 0'],one_storm_EU, label='Max of the path in EU borders')#, lw=0, marker='o')
        plt.plot(one_storm_mean['Unnamed: 0'], one_storm_mean['0'], label='Mean at level '+level+' hPa')
        plt.scatter(one_storm_EU_mean['Unnamed: 0'], one_storm_EU_mean['0'], label='Mean at level '+level+' of the path in EU borders')
        #plt.plot(one_storm_min['Unnamed: 0'], one_storm_min['0'], label='Min of the whole path')
        #plt.scatter(one_storm_EU_min['Unnamed: 0'], one_storm_EU_min['0'], label='Min of the path in EU borders')
        #plt.plot(one_storm_std['Unnamed: 0'], one_storm_std['0'], label='Std of the whole path')
        #plt.scatter(one_storm_EU_std['Unnamed: 0'], one_storm_EU_std['0'], label='Std of the path in EU borders')


    plt.title(f'Comparison of the different datasets for relative humidity in storm {storm}')
    plt.xlabel('Step')
    plt.ylabel('RH in %')
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


# %%
'''# test 2

# open each csv file and read it into a pandas dataframe
name_of_variables = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/variable_list_24_v2_1.csv')
name_of_variables = name_of_variables['variable'].tolist()

# Create a new list of variables
new_name_of_variables = []
for variable in name_of_variables:
    if variable == 'geopotential':
        new_name_of_variables.extend(['geopotential_500', 'geopotential_1000'])
    else:
        new_name_of_variables.append(variable)

# Iterate over the new list of variables
for i in range(0, len(new_name_of_variables)):

    print(new_name_of_variables[i])

    locals()[f'max_{new_name_of_variables[i]}'] = pd.DataFrame()
    locals()[f'min_{new_name_of_variables[i]}'] = pd.DataFrame()
    locals()[f'mean_{new_name_of_variables[i]}'] = pd.DataFrame()
    locals()[f'sigma_{new_name_of_variables[i]}'] = pd.DataFrame()

    for j in range (1,96+1):
        if 'geopotential' in new_name_of_variables[i]:
            for k in [500, 1000]:
                df_max_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/geopotential/storm_' + str(j) + '/max_'+ str(j) + '_' + str(k) + '.csv')
                df_min_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/geopotential/storm_' + str(j) + '/min_'+ str(j) + '_' + str(k) + '.csv')
                df_mean_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/geopotential/storm_' + str(j) + '/mean_'+ str(j) + '_' + str(k) + '.csv')
                df_sigma_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/geopotential/storm_' + str(j) + '/std_'+ str(j) + '_' + str(k) + '.csv')

        else:
            df_max_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/' + new_name_of_variables[i] + '/storm_' + str(j) + '/max_'+ str(j) + '_0.csv')
            df_min_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/' + new_name_of_variables[i] + '/storm_' + str(j) + '/min_'+ str(j) + '_0.csv')
            df_mean_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/' + new_name_of_variables[i] + '/storm_' + str(j) + '/mean_'+ str(j) + '_0.csv')
            df_sigma_temp = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/datasets_3h/' + new_name_of_variables[i] + '/storm_' + str(j) + '/std_'+ str(j) + '_0.csv')

        df_max_temp = df_max_temp.drop(columns = ['Unnamed: 0'])
        df_min_temp = df_min_temp.drop(columns = ['Unnamed: 0'])
        df_mean_temp = df_mean_temp.drop(columns = ['Unnamed: 0'])
        df_sigma_temp = df_sigma_temp.drop(columns = ['Unnamed: 0'])

        locals()[f'max_{new_name_of_variables[i]}'] = pd.concat([locals()[f'max_{new_name_of_variables[i]}'], df_max_temp], axis=0)
        locals()[f'min_{new_name_of_variables[i]}'] = pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=0)
        locals()[f'mean_{new_name_of_variables[i]}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=0)
        locals()[f'sigma_{new_name_of_variables[i]}'] = pd.concat([locals()[f'sigma_{new_name_of_variables[i]}'], df_sigma_temp], axis=0)'''

# %%
'''# to separate each storm into one column : 

# open each csv file and read it into a pandas dataframe

name_of_variables = pd.read_csv('/Users/fabienaugsburger/Documents/GitHub/master-project/variable_list_24_v2_1.csv')
name_of_variables = name_of_variables['variable'].tolist()

# which dataset to use 

dataset = 'datasets_1h'

# Create a new list of variables
new_name_of_variables = []
for variable in name_of_variables:
    if variable == 'geopotential':
        new_name_of_variables.extend(['geopotential_500', 'geopotential_1000'])
    #if  variable == 'relative_humidity' or variable == 'vertical_velocity':
        #for level in full_pressure:
            #new_name_of_variables.append(variable + '_' + level)
    else:
        new_name_of_variables.append(variable)

# Iterate over the new list of variables
for i in range(0, len(new_name_of_variables)):

    print(new_name_of_variables[i])

    locals()[f'max_{new_name_of_variables[i]}'] = pd.DataFrame()
    locals()[f'min_{new_name_of_variables[i]}'] = pd.DataFrame()
    locals()[f'mean_{new_name_of_variables[i]}'] = pd.DataFrame()
    locals()[f'sigma_{new_name_of_variables[i]}'] = pd.DataFrame()

    for j in range(1, 95 + 1):
        try:
            if 'geopotential' in variable or 'relative_humidity' in variable or 'vertical_velocity' in variable:
                continue
                for k in [500, 1000]:
                    df_max_temp = pd.read_csv(f'{path}{dataset}/geopotential/storm_{j}/max_{j}_{k}.csv')
                    df_min_temp = pd.read_csv(f'{path}{dataset}/geopotential/storm_{j}/min_{j}_{k}.csv')
                    df_mean_temp = pd.read_csv(f'{path}{dataset}/geopotential/storm_{j}/mean_{j}_{k}.csv')
                    df_sigma_temp = pd.read_csv(f'{path}{dataset}/geopotential/storm_{j}/std_{j}_{k}.csv')

            else:
                df_max_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/max_{j}_0.csv')
                df_min_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/min_{j}_0.csv')
                df_mean_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/mean_{j}_0.csv')
                df_sigma_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/std_{j}_0.csv')
                #print(f'File for storm {j} and variable {variable} found.')
        except FileNotFoundError:
            print(f"File for storm {j} and variable {variable} not found.")
            continue

        df_max_temp = df_max_temp.drop(columns = ['Unnamed: 0'])
        df_min_temp = df_min_temp.drop(columns = ['Unnamed: 0'])
        df_mean_temp = df_mean_temp.drop(columns = ['Unnamed: 0'])
        df_sigma_temp = df_sigma_temp.drop(columns = ['Unnamed: 0'])

        df_max_temp = df_max_temp.rename(columns = {'0': f'storm_{j}'})
        df_min_temp = df_min_temp.rename(columns = {'0': f'storm_{j}'})
        df_mean_temp = df_mean_temp.rename(columns = {'0': f'storm_{j}'})
        df_sigma_temp = df_sigma_temp.rename(columns = {'0': f'storm_{j}'})

        locals()[f'max_{new_name_of_variables[i]}']= pd.concat([locals()[f'max_{new_name_of_variables[i]}'] ,df_max_temp], axis=1)#[f'storm_{j}'] = df_max_temp['0']
        locals()[f'min_{new_name_of_variables[i]}']= pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=1)#[f'storm_{j}'] = df_min_temp['0']# = pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=0)
        locals()[f'mean_{new_name_of_variables[i]}']= pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=1)#[f'storm_{j}'] = df_mean_temp['0'] #= pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=0)
        locals()[f'sigma_{new_name_of_variables[i]}']= pd.concat([locals()[f'sigma_{new_name_of_variables[i]}'], df_sigma_temp], axis=1)#[f'storm_{j}'] = df_sigma_temp['0'] #= pd.concat([locals()[f'sigma_{new_name_of_variables[i]}'], df_sigma_temp], axis=0)

for k in [500, 1000]:
    locals()[f'max_geopotential_{k}'] = pd.DataFrame()
    locals()[f'min_geopotential_{k}'] = pd.DataFrame()
    locals()[f'mean_geopotential_{k}'] = pd.DataFrame()
    locals()[f'sigma_geopotential_{k}'] = pd.DataFrame()
    for j in range (1,95+1):
        try :
            df_max_temp = pd.read_csv(path+dataset+'/geopotential/storm_' + str(j) + '/max_'+ str(j) + '_' + str(k) + '.csv')
            df_min_temp = pd.read_csv(path+dataset+'/geopotential/storm_' + str(j) + '/min_'+ str(j) + '_' + str(k) + '.csv')
            df_mean_temp = pd.read_csv(path+dataset+'/geopotential/storm_' + str(j) + '/mean_'+ str(j) + '_' + str(k) + '.csv')
            df_sigma_temp = pd.read_csv(path+dataset+'/geopotential/storm_' + str(j) + '/std_'+ str(j) + '_' + str(k) + '.csv')
        except FileNotFoundError:
            print(f"Fichier pour la tempête {j} et la variable geopotential_{k} non trouvé.")
            continue

        df_max_temp = df_max_temp.drop(columns = ['Unnamed: 0'])
        df_min_temp = df_min_temp.drop(columns = ['Unnamed: 0'])
        df_mean_temp = df_mean_temp.drop(columns = ['Unnamed: 0'])
        df_sigma_temp = df_sigma_temp.drop(columns = ['Unnamed: 0'])

        df_max_temp = df_max_temp.rename(columns = {'0': f'storm_{j}'})
        df_min_temp = df_min_temp.rename(columns = {'0': f'storm_{j}'})
        df_mean_temp = df_mean_temp.rename(columns = {'0': f'storm_{j}'})
        df_sigma_temp = df_sigma_temp.rename(columns = {'0': f'storm_{j}'})

        locals()[f'max_geopotential_{k}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_max_temp], axis=1)#[f'storm_{j}'] = df_max_temp['0']
        locals()[f'min_geopotential_{k}'] = pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=1)#[f'storm_{j}'] = df_min_temp['0']# = pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=0)
        locals()[f'mean_geopotential_{k}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=1)#[f'storm_{j}'] = df_mean_temp['0'] #= pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=0)
        locals()[f'sigma_geopotential_{k}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_sigma_temp], axis=1)#[f'storm_{j}'] = df_sigma_temp['0']
PL_var = ['relative_humidity', 'vertical_velocity']
for variable in PL_var:
    for level in full_pressure:
            locals()[f'max_{variable}_{level}'] = pd.DataFrame()
            locals()[f'min_{variable}_{level}'] = pd.DataFrame()
            locals()[f'mean_{variable}_{level}'] = pd.DataFrame()
            locals()[f'sigma_{variable}_{level}'] = pd.DataFrame()
            for j in range (1,95+1):
                try :
                    df_max_temp = pd.read_csv(path+dataset+'/{variable}/storm_' + str(j) + '/max_'+ str(j) + '_' + str(level) + '.csv')
                    df_min_temp = pd.read_csv(path+dataset+'/{variable}/storm_' + str(j) + '/min_'+ str(j) + '_' + str(level) + '.csv')
                    df_mean_temp = pd.read_csv(path+dataset+'/{variable}/storm_' + str(j) + '/mean_'+ str(j) + '_' + str(level) + '.csv')
                    df_sigma_temp = pd.read_csv(path+dataset+'/{variable}/storm_' + str(j) + '/std_'+ str(j) + '_' + str(level) + '.csv')
                except FileNotFoundError:
                    print(f"Fichier pour la tempête {j} et la variable {variable}_{level} non trouvé.")
                    continue

                df_max_temp = df_max_temp.drop(columns = ['Unnamed: 0'])
                df_min_temp = df_min_temp.drop(columns = ['Unnamed: 0'])
                df_mean_temp = df_mean_temp.drop(columns = ['Unnamed: 0'])
                df_sigma_temp = df_sigma_temp.drop(columns = ['Unnamed: 0'])

                df_max_temp = df_max_temp.rename(columns = {'0': f'storm_{j}'})
                df_min_temp = df_min_temp.rename(columns = {'0': f'storm_{j}'})
                df_mean_temp = df_mean_temp.rename(columns = {'0': f'storm_{j}'})
                df_sigma_temp = df_sigma_temp.rename(columns = {'0': f'storm_{j}'})

                locals()[f'max_{variable}_{level}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_max_temp], axis=1)#[f'storm_{j}'] = df_max_temp['0']
                locals()[f'min_{variable}_{level}'] = pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=1)#[f'storm_{j}'] = df_min_temp['0']# = pd.concat([locals()[f'min_{new_name_of_variables[i]}'], df_min_temp], axis=0)
                locals()[f'mean_{variable}_{level}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=1)#[f'storm_{j}'] = df_mean_temp['0'] #= pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_mean_temp], axis=0)
                locals()[f'sigma_{variable}_{level}'] = pd.concat([locals()[f'mean_{new_name_of_variables[i]}'], df_sigma_temp], axis=1)#[f'storm_{j}'] = df_sigma_temp['0']'''

# %%
import pandas as pd

def split_variable_level(variable_with_level):
    parts = variable_with_level.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    else:
        return variable_with_level, 0

# Read the list of variables
name_of_variables = pd.read_csv(f'{path}variable_list_24_v5.csv')
#name_of_variables = name_of_variables['variable'].tolist()

# Define the dataset to use
dataset = 'datasets_1h'
missing = []

# Create a new list of variables
#new_name_of_variables = []
for idx, row in name_of_variables.iterrows():
    variable = split_variable_level(row['variables'])[0]
    level = split_variable_level(row['variables'])[1]
    print(variable, level)

    locals()[f'max_{variable}_{level}'] = pd.DataFrame()
    locals()[f'min_{variable}_{level}'] = pd.DataFrame()
    locals()[f'mean_{variable}_{level}'] = pd.DataFrame()
    locals()[f'sigma_{variable}_{level}'] = pd.DataFrame()

    for j in range(1, 96 + 1):
    #if level == 'pl':
        #for lvl in level:
            try:
                    df_max_temp = pd.read_csv(f'{path}DATASETS/{dataset}/{variable}/storm_{j}/max_{j}_{level}.csv')
                    df_min_temp = pd.read_csv(f'{path}DATASETS/{dataset}/{variable}/storm_{j}/min_{j}_{level}.csv')
                    df_mean_temp = pd.read_csv(f'{path}DATASETS/{dataset}/{variable}/storm_{j}/mean_{j}_{level}.csv')
                    df_sigma_temp = pd.read_csv(f'{path}DATASETS/{dataset}/{variable}/storm_{j}/std_{j}_{level}.csv')
                    #print(f'File for storm {j} and variable {variable} found.')
            except FileNotFoundError:
                ds_missing = missing.append(f'{variable}_{level}')
                #print(f"File for storm {j}, variable {variable}, and level {level} not found.")
                continue
                    # Check if the columns exist before dropping/renaming
            if 'Unnamed: 0' in df_max_temp.columns:
                df_max_temp = df_max_temp.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_min_temp.columns:
                df_min_temp = df_min_temp.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_mean_temp.columns:
                df_mean_temp = df_mean_temp.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_sigma_temp.columns:
                df_sigma_temp = df_sigma_temp.drop(columns=['Unnamed: 0'])

            if '0' in df_max_temp.columns:
                df_max_temp = df_max_temp.rename(columns={'0': f'storm_{j}'})
            if '0' in df_min_temp.columns:
                df_min_temp = df_min_temp.rename(columns={'0': f'storm_{j}'})
            if '0' in df_mean_temp.columns:
                df_mean_temp = df_mean_temp.rename(columns={'0': f'storm_{j}'})
            if '0' in df_sigma_temp.columns:
                df_sigma_temp = df_sigma_temp.rename(columns={'0': f'storm_{j}'})

            locals()[f'max_{variable}_{level}'] = pd.concat([locals()[f'max_{variable}_{level}'], df_max_temp], axis=1)
            locals()[f'min_{variable}_{level}'] = pd.concat([locals()[f'min_{variable}_{level}'], df_min_temp], axis=1)
            locals()[f'mean_{variable}_{level}'] = pd.concat([locals()[f'mean_{variable}_{level}'], df_mean_temp], axis=1)
            locals()[f'sigma_{variable}_{level}'] = pd.concat([locals()[f'sigma_{variable}_{level}'], df_sigma_temp], axis=1)

'''        else:
            try:
                    df_max_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/max_{j}_0.csv')
                    df_min_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/min_{j}_0.csv')
                    df_mean_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/mean_{j}_0.csv')
                    df_sigma_temp = pd.read_csv(f'{path}{dataset}/{variable}/storm_{j}/std_{j}_0.csv')
                    #print(f'File for storm {j} and variable {variable} found.')
            except FileNotFoundError:
                ds_missing = missing.append(f'{variable}_{level}')
                #print(f"File for storm {j}, variable {variable}, and level {level} not found.")
                continue
            # Check if the columns exist before dropping/renaming
            if 'Unnamed: 0' in df_max_temp.columns:
                df_max_temp = df_max_temp.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_min_temp.columns:
                df_min_temp = df_min_temp.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_mean_temp.columns:
                df_mean_temp = df_mean_temp.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in df_sigma_temp.columns:
                df_sigma_temp = df_sigma_temp.drop(columns=['Unnamed: 0'])

            if '0' in df_max_temp.columns:
                df_max_temp = df_max_temp.rename(columns={'0': f'storm_{j}_level_0'})
            if '0' in df_min_temp.columns:
                df_min_temp = df_min_temp.rename(columns={'0': f'storm_{j}_level_0'})
            if '0' in df_mean_temp.columns:
                df_mean_temp = df_mean_temp.rename(columns={'0': f'storm_{j}_level_0'})
            if '0' in df_sigma_temp.columns:
                df_sigma_temp = df_sigma_temp.rename(columns={'0': f'storm_{j}_level_0'})

            locals()[f'max_{variable}'] = pd.concat([locals()[f'max_{variable}'], df_max_temp], axis=1)
            locals()[f'min_{variable}'] = pd.concat([locals()[f'min_{variable}'], df_min_temp], axis=1)
            locals()[f'mean_{variable}'] = pd.concat([locals()[f'mean_{variable}'], df_mean_temp], axis=1)
            locals()[f'sigma_{variable}'] = pd.concat([locals()[f'sigma_{variable}'], df_sigma_temp], axis=1)'''

# %%
# check shape of the dataframes, they should be (472, 96)
stats = ['max', 'min', 'mean', 'sigma']

for idx, row in name_of_variables.iterrows():
    variable = split_variable_level(row['variables'])[0]
    level = split_variable_level(row['variables'])[1]
    for key in stats:
        if dataset == 'datasets_1h':
            if locals()[f'{key}_{variable}_{level}'].shape != (472, 96):
                print(variable, level)
        else:
            if locals()[f'{key}_{variable}_{level}'].shape != (158, 96):
                print(variable, level)
    '''print(variable, level)
    print(locals()[f'max_{variable}_{level}'].shape)
    print(locals()[f'min_{variable}_{level}'].shape)
    print(locals()[f'mean_{variable}_{level}'].shape)
    print(locals()[f'sigma_{variable}_{level}'].shape)'''

# %%
# store the mean_large_scale_precipitation in a df, set the name as the header of the column and so the same with mean_total_precipitation

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)                       
setup_seed(42)

number_test_storms = round(96*0.3)
test_storm_index = random.sample(range(96), number_test_storms)
number_validation_storms = round(number_test_storms*1/3)
validation_storm_index = random.sample(test_storm_index, number_validation_storms)

## for later, pick defined storms for test and validation set rather than random but ranbdomized the seed to have a different random selection

# order the index of the test_storm_index and validation_storm_index
test_storm_index.sort()
validation_storm_index.sort()

# remove the index of the validation storms from the test_storm_index
'''for i in range(0, len(validation_storm_index)):
    test_storm_index.remove(validation_storm_index[i])'''

# check the number of storms in each set
print(len(test_storm_index))
print(len(validation_storm_index))

# create the training, validation and test set
X_train = pd.DataFrame()
X_validation = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_validation = pd.DataFrame()
y_test = pd.DataFrame()

# remove the instantaneous_10m_wind_gust from the list of variables
name_of_variables_X = name_of_variables.copy()
name_of_variables_X = name_of_variables_X[name_of_variables_X['variables'] != 'instantaneous_10m_wind_gust']

for idx, row in name_of_variables_X.iterrows(): 
    x_var = split_variable_level(row['variables'])[0]
    level = split_variable_level(row['variables'])[1]
    for storm_number in locals()[f'mean_{x_var}_{level}'].columns:
        modified_storm_number = f"{x_var}_{storm_number}_level_{level}"
        if int(storm_number.split('_')[1]) in test_storm_index:
            if int(storm_number.split('_')[1]) in validation_storm_index:
                X_validation = pd.concat([X_validation, locals()[f'mean_{x_var}_{level}'][storm_number]], axis=1) # [storm_number]
                X_validation = X_validation.rename(columns={storm_number: modified_storm_number})
                #y_validation = pd.concat([y_validation, mean_instantaneous_10m_wind_gust[storm_number]], axis=1)
            else:
                X_test = pd.concat([X_test, locals()[f'mean_{x_var}_{level}'][storm_number]], axis=1)
                X_test = X_test.rename(columns={storm_number: modified_storm_number})
                #y_test = pd.concat([y_test, locals()[f'mean_instantaneous_10m_wind_gust'][storm_number]], axis=1)
        else:
            X_train = pd.concat([X_train, locals()[f'mean_{x_var}_{level}'][storm_number]], axis=1)
            X_train = X_train.rename(columns={storm_number: modified_storm_number})
            #y_train = pd.concat([y_train, locals()[f'mean_instantaneous_10m_wind_gust'][storm_number]], axis=1)

for storm_number in mean_instantaneous_10m_wind_gust_0.columns:
    modified_storm_number = f"mean_instantaneous_10m_wind_gust_{storm_number}_level_0"
    if int(storm_number.split('_')[1]) in test_storm_index:
        if int(storm_number.split('_')[1]) in validation_storm_index:
            y_validation = pd.concat([y_validation, mean_instantaneous_10m_wind_gust_0[storm_number]], axis=1)
            y_validation = y_validation.rename(columns={storm_number: modified_storm_number})
        else:
            y_test = pd.concat([y_test, mean_instantaneous_10m_wind_gust_0[storm_number]], axis=1)
            y_test = y_test.rename(columns={storm_number: modified_storm_number})
    else:
        y_train = pd.concat([y_train, mean_instantaneous_10m_wind_gust_0[storm_number]], axis=1)
        y_train = y_train.rename(columns={storm_number: modified_storm_number})

# %%
print(f'Storms in testing set: {test_storm_index}, and in validation set: {validation_storm_index}')

# %%
import pandas as pd
import re

def concatenate_storms(df):
    # Extract unique storms and levels from column names
    storms = set(re.findall(r'storm_(\d+)', ' '.join(df.columns)))
    levels = set(re.findall(r'level_(\d+)', ' '.join(df.columns)))
    
    concatenated_data = []

    for storm in storms:
        storm_df_list = []
        for level in levels:
            # Filter columns for the specific storm and level
            level_columns = [col for col in df.columns if f'storm_{storm}_level_{level}' in col]
            if level_columns:
                level_df = df[level_columns]
                # Rename columns to include the level
                level_df.columns = [re.sub(r'_storm_\d+_level_', '_', col) for col in level_df.columns]
                # Drop rows with all NaN values
                level_df = level_df.dropna(how='all')
                storm_df_list.append(level_df)

        # Concatenate all levels for this storm along axis 1
        if storm_df_list:
            storm_df = pd.concat(storm_df_list, axis=1)
            concatenated_data.append(storm_df)
    
    # Concatenate all storms along axis 0
    final_df = pd.concat(concatenated_data, axis=0).reset_index(drop=True)

    # Remove duplicated columns
    final_df = final_df.loc[:,~final_df.T.duplicated()]
    
    return final_df

# Example usage:
# df = pd.read_csv('your_data.csv')  # Replace with your DataFrame loading method

X_train_new= concatenate_storms(X_train)
X_test_new= concatenate_storms(X_test)
X_validation_new= concatenate_storms(X_validation)

y_train_new= concatenate_storms(y_train)
y_test_new= concatenate_storms(y_test)
y_validation_new= concatenate_storms(y_validation)

# final_df.to_csv('concatenated_data.csv', index=False)  # Replace with your DataFrame saving method

# %%
#input_shape = (X_train_new.shape)
#input_shape

# save the dataframes with pickle
'''X_train_new.to_pickle('/Users/fabienaugsburger/Documents/GitHub/master-project/X_train.pkl')
X_test_new.to_pickle('/Users/fabienaugsburger/Documents/GitHub/master-project/X_test.pkl')
X_validation_new.to_pickle('/Users/fabienaugsburger/Documents/GitHub/master-project/X_validation.pkl')
y_train_new.to_pickle('/Users/fabienaugsburger/Documents/GitHub/master-project/y_train.pkl')
y_test_new.to_pickle('/Users/fabienaugsburger/Documents/GitHub/master-project/y_test.pkl')
y_validation_new.to_pickle('/Users/fabienaugsburger/Documents/GitHub/master-project/y_validation.pkl')'''
path_pk = f'{path}DATASETS/X_y/old/'

# load the dataframes with pickle
X_train_new = pd.read_pickle(f'{path_pk}/X_train.pkl')
X_test_new = pd.read_pickle(f'{path_pk}/X_test.pkl')
X_validation_new = pd.read_pickle(f'{path_pk}/X_validation.pkl')
y_train_new = pd.read_pickle(f'{path_pk}/y_train.pkl')
y_test_new = pd.read_pickle(f'{path_pk}/y_test.pkl')
y_validation_new = pd.read_pickle(f'{path_pk}/y_validation.pkl')

# %%
# Define the model
model_Gia_2 = Sequential()

# Define the input shape in the first layer of the neural network
input_shape = (X_train_new.shape[1], 1)

# Add a Conv1D layer with 32 filters and a kernel size of 3
#model_Gia.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))

# Add the first LSTM layer with 128 units
model_Gia_2.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))

# Add the second LSTM layer with 256 units
#model_Gia_2.add(LSTM(units=256))

# Add a Dense (fully connected) layer with 128 units and tanh activation
model_Gia_2.add(Dense(units=128, activation='tanh'))

# Flatten the output
model_Gia_2.add(Flatten())

# Add the final Dense layer with a linear activation function
# Assuming the output length is predefined, for example, 1
output_length = 1
model_Gia_2.add(Dense(units=output_length, activation='linear'))

# Compile the model
model_Gia_2.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model_Gia_2.summary()

# %%
run_index = 3 # it should be an integer, e.g. 1

run_logdir = os.path.join(os.curdir, "LSTM_Giaremis_logs/", "run_{:03d}".format(run_index))

print(run_logdir)

# %%
# Define callbacks (they can really improve the accuracy if well-chosen!)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_Gia_2.keras", 
                                                   save_best_only=True,
                                                   monitor='loss')
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# %%
# Définir le modèle
'''X = df #pd.concat([locals()[f'{x_var}']['0'] for x_var in variable_w_high_corr if f'{x_var}' in locals()], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, target_variable, test_size=0.2, random_state=42)
'''
model = Sequential()
# Ajouter le nombre de variables à haute corrélation comme variable indépendante
model.add(LSTM(50, activation='relu', input_shape=(X_train_new.shape[1], 1)))
model.add(Dense(1))

# Diviser les données en ensembles de formation et de test
#X_train, X_test, y_train, y_test = train_test_split(X, target_variable, test_size=0.2, random_state=42)

# Compiler le modèle
model.compile(optimizer='adam', loss='mse')

# Redimensionner les données pour LSTM
'''if len(X_train.shape) == 2:
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1]))
    y_train = y_train.values.reshape((y_train.shape[0], y_train.shape[1]))
    y_test = y_test.values.reshape((y_test.shape[0], y_test.shape[1]))'''

# Entraîner le modèle
model.fit(X_train_new, y_train_new, 
          epochs=200, 
          verbose=1, 
          use_multiprocessing=True, 
          validation_data=(X_validation_new, y_validation_new), 
          callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

# Faire des prédictions
y_pred = model.predict(X_test_new)

# %%
# fit the Giaremis model
# Enable mixed precision policy (optional but recommended for performance)
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
# Entraîner le modèle
model_Gia_2.fit(X_train_new, y_train_new, 
          epochs=10, 
          verbose=1,  
          batch_size=32,
          callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb],)
          #use_multiprocessing=True)

# Faire des prédictions
y_pred = model_Gia_2.predict(X_test_new)

# %%
#%tensorboard --logdir=./LSTM_basic_logs --port=1234
# to launch tensorboard in the terminal
#tensorboard --logdir=/Users/fabienaugsburger/Documents/GitHub/master-project/case_study/LSTM_basic_logs --port=1234

# %%
new = pd.DataFrame(y_pred)
new = new.rename(columns={0: 'y_pred'})
new = new.reset_index(drop=True)
new = new.set_index(X_test_new.index)

# plot the y_pred and y_test

plt.plot(new['y_pred'], label='y_pred')
plt.plot(y_test_new, label='y_test')
plt.legend()
plt.show()

# calculate the RMSE

rms = (mean_squared_error(y_test_new, new['y_pred']))**0.5
print(rms) 

# %%
y_pred_train = model_Gia_2.predict(X_train_new)

new_train = pd.DataFrame(y_pred_train)
new_train = new_train.rename(columns={0: 'y_pred'})
new_train = new_train.reset_index(drop=True)
new_train = new_train.set_index(X_train_new.index)

# plot the y_pred and y_test

plt.plot(new_train['y_pred'], label='y_pred')
plt.plot(y_train_new, label='y_train')
plt.legend()
plt.show()

# %%
# Create a SHAP explainer TAKES 1 HOUR TO RUN
explainer = shap.KernelExplainer(model_tuned.predict, X_test_new[:100])  # Use a small subset for speed model_Gia_2

# Compute SHAP values for a subset of the data
shap_values = explainer.shap_values(X_test_new[:100])

# Plot SHAP summary
shap.summary_plot(shap_values, X_test_new[:100])

# %%
# Plot SHAP dependence

shap_values_pd = pd.DataFrame(shap_values[1,:][:])
shap_values_pd = shap_values_pd.rename(columns={0: 'shap_values'})

# rename the index by the variables' names
shap_values_pd = shap_values_pd.set_index(X_test_new[:100].columns)
shap_values_pd
shap_values_graph = shap_values_pd['shap_values']
index = shap_values_pd.index

# Create a new figure
plt.figure()

# Swap the x and y values
plt.plot(shap_values_graph, index, label='shap_values', )

# Set labels
plt.xlabel('SHAP Values')
plt.ylabel('Feature Index')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# %%
# Assuming shap_values and X_train_new are defined
# Convert SHAP values and feature names to DataFrame for easier handling
shap_values_2d = np.reshape(shap_values, (shap_values.shape[0], shap_values.shape[1]))

shap_values_df = pd.DataFrame(shap_values_2d, columns=[f"Feature {i}" for i in range(X_train_new.shape[1])])
feature_names = X_test_new[:100].columns

# Compute mean absolute SHAP values for each feature
mean_abs_shap_values = shap_values_df.abs().mean(axis=0)

# Create a DataFrame for easier plotting
shap_summary = pd.DataFrame({
    'Feature': feature_names,
    'Mean Absolute SHAP Value': mean_abs_shap_values
})

# Sort features by mean absolute SHAP value
shap_summary = shap_summary.sort_values(by='Mean Absolute SHAP Value', ascending=False)

# Plot horizontal bar chart
plt.figure(figsize=(15, 15))
plt.barh(shap_summary['Feature'], shap_summary['Mean Absolute SHAP Value'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Feature Importance based on SHAP Values')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
plt.show()

# %%
# Save the explainer to a file
with open('shap_explainer_test.pkl', 'wb') as f:
    pickle.dump(explainer, f)

# Save the SHAP values if needed
with open('shap_values_test.pkl', 'wb') as f:
    pickle.dump(shap_values, f)

# %%
# Assuming model.predict returns the output for each sample
predictions = model_Gia_2.predict(X_train_new)

# Using permutation importance
perm_importance = permutation_importance(model_Gia_2, X_train_new, y_train_new, n_repeats=10, random_state=42)

# Plotting permutation importances

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(range(X_train_new.shape[2]), perm_importance.importances_mean[sorted_idx])
plt.yticks(range(X_train_new.shape[2]), np.array(["Feature " + str(i) for i in sorted_idx]))
plt.xlabel("Permutation Importance")
plt.show()

# %%
# Define a function to shuffle feature values
def permute_feature(X, feature_index):
    X_permuted = X.copy()
    np.random.shuffle(X_permuted[:, :, feature_index])
    return X_permuted

# Calculate baseline performance
baseline_performance = mean_squared_error(y_train_new, model_Gia_2.predict(X_train_new))

# Initialize an empty list to hold feature importances
feature_importances = []

# Loop over each feature and calculate importance
for i in range(X_train_new.shape[2]):
    X_permuted = permute_feature(X_train, i)
    permuted_performance = mean_squared_error(y_train, model.predict(X_permuted))
    feature_importances.append(permuted_performance - baseline_performance)

# Plot feature importances
plt.barh(range(X_train_new.shape[2]), feature_importances)
plt.yticks(range(X_train_new.shape[2]), np.array(["Feature " + str(i) for i in range(X_train.shape[2])]))
plt.xlabel("Feature Permutation Importance")
plt.show()

# %% [markdown]
# Hyperparameter tuning, using GPT, without adaptation

# %%
import optuna
from tensorflow.keras.optimizers import Adam


def create_model(trial):
    # Suggest hyperparameters
    n_units = trial.suggest_int('n_units', 50, 300)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=n_units, return_sequences=True, input_shape=(X_train_new.shape[1], 1))) #X_train_new.shape[2]
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Additional LSTM layers
    for _ in range(n_layers - 1):
        model.add(LSTM(units=n_units, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Flatten and add Dense layers
    model.add(Flatten())
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=1, activation='linear'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    return model

def objective(trial):
    model = create_model(trial)
    
    # Train the model
    history = model.fit(X_train_new, y_train_new, 
                        validation_data=(X_validation_new, y_validation_new),
                        epochs=10,
                        batch_size=32,
                        verbose=1)
    
    # Evaluate the model
    loss = model.evaluate(X_validation_new, y_validation_new, verbose=0)
    
    return loss

# %%
# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, 
               n_trials=50,
               n_jobs=-1)  # Use all available CPUs

# Get the best hyperparameters
print('Best hyperparameters: ', study.best_params)

# %%
# Retrieve the best hyperparameters
best_params = study.best_params

# Create the final model with the best hyperparameters
final_model = create_model(optuna.trial.FixedTrial(best_params))

# Train the final model
final_model.fit(X_train_new, y_train_new, 
                validation_data=(X_validation_new, y_validation_new),
                epochs=10,
                batch_size=32)

# %%
# save the model with keras 
final_model.save('tuned_model_Gia_2.keras')
'''with open('tuned_model_Gia_2.pkl', 'wb') as f:
    pickle.dump(final_model, f)'''

# %%
# load model_win.h5

#model_tuned = pickle.load(open(f'{path}case_study/tuned_model_Gia_2.pkl', 'rb'))
model_tuned = pickle.load(open('C:/Users/fabau/OneDrive/Documents/GitHub/master-project/case_study/tuned_model_Gia_2.pkl', 'rb'))

#model = tf.keras.models.load_model('/Users/fabienaugsburger/Documents/GitHub/master-project/case_study/model_win.h5')

# %%
# use shapely to calculate the shap values
# Sample a subset of the data
sample_size = 1  # Adjust this to a size your machine can handle
X_sample = X_test_new.sample(sample_size, random_state=42)
new_sample = new.loc[X_sample.index]

# see the explanation for the model's predictions using SHAP
#explainer = shap.DeepExplainer(model, X_test_new)
explainer = shap.KernelExplainer(model_Gia_2, X_sample, feature_names=X_sample.columns)
shap_values = explainer(X_sample)

shap.plots.bar(shap_values[0])


# %%
for i in range(0, len(new_name_of_variables)):
    shap.plots.bar(shap_values[i], max_display=10)


# %%
# Utilisation de DeepExplainer pour expliquer les prédictions du modèle
background = X_sample
explainer = shap.KernelExplainer(model, background)
shap_values = explainer.shap_values(X_sample)

# Utilisation de shap.plots.bar pour visualiser l'importance des features
shap.plots.bar(shap_values[0])

# %%
shape_values(X_sample, npermutations=100)   

# %%
# look at the shap values for the first prediction
print(np.asarray(shap_values[:].values).shape)
test = np.squeeze(np.asarray(shap_values[:].values))
test_m = np.mean(test, axis=0)
#shap.plots.waterfall(shap_values[])


# %%
# plot the results 

plt.plot(np.squeeze(y_train_new), label='True', lw=0, marker='x')
plt.plot(new_train, label='Predicted')
plt.ylim(0, 50)
plt.xlim(0,2000)
plt.legend()
plt.show()

# %%
# feature importance 

result = permutation_importance(model, X_train_new, y_train_new, n_repeats=10, random_state=42, n_jobs=-1, scoring='neg_root_mean_squared_error')



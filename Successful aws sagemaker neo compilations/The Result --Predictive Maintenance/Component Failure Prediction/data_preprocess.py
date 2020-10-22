import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from datetime import datetime 
from datetime import timedelta

def telemetry_preprocess(telemetry):
    # format datetime field which comes in as string
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

    print("Total number of telemetry records: %d" % len(telemetry.index))
#     print(telemetry.head()) 
    return(telemetry)
    
    
def error_preprocess(errors):
    errors['datetime'] = pd.to_datetime(errors['datetime'],format = '%Y-%m-%d %H:%M:%S')
    errors['errorID'] = errors['errorID'].astype('category')
    print("Total Number of error records: %d" %len(errors.index))
    return(errors)
    
def maint_preprocess(maint):
    
    maint['datetime'] = pd.to_datetime(maint['datetime'], format='%Y-%m-%d %H:%M:%S')
    maint['comp'] = maint['comp'].astype('category')
    print("Total Number of maintenance Records: %d" %len(maint.index))
    return(maint)

def mach_preprocess(machines):
    machines['model'] = machines['model'].astype('category')

    print("Total number of machines: %d" % len(machines.index))
    return(machines)
    
    
def failure_preprocess(failures):
    # format datetime field which comes in as string
    failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
    failures['failure'] = failures['failure'].astype('category')

    print("Total number of failures: %d" % len(failures.index))
    return(failures)
    
def get_lagfeatures_tele(telemetry):
    # Calculate mean values for telemetry features
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right').mean().unstack())
    telemetry_mean_3h = pd.concat(temp, axis=1)
    telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace=True)
    print(telemetry_mean_3h)
    # repeat for standard deviation
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right').std().unstack())
    telemetry_sd_3h = pd.concat(temp, axis=1)
    telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace=True)
    print(telemetry_sd_3h)
    # Calculate mean values for telemetry features
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.DataFrame.rolling(pd.pivot_table(telemetry,
                                                   index='datetime',
                                                   columns='machineID',
                                                   values=col), window=24).mean().resample('3H',
                                                                                    closed='left',
                                                                                    label='right').first().unstack())

    #print(temp)    
    telemetry_mean_24h = pd.concat(temp, axis=1)
    telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace=True)
    telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]
    print(telemetry_mean_24h)
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.DataFrame.rolling(pd.pivot_table(telemetry,
                                                   index='datetime',
                                                   columns='machineID',
                                                   values=col), window=24).std().resample('3H',
                                                                                    closed='left',
                                                                                    label='right').first().unstack())

    telemetry_sd_24h = pd.concat(temp, axis=1)
    telemetry_sd_24h.columns = [i + 'std_24h' for i in fields]
    telemetry_sd_24h.reset_index(inplace=True)
    telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltstd_24h'].isnull()]
    print(telemetry_sd_24h)
    # merge columns of feature sets created earlier
    telemetry_feat = pd.concat([telemetry_mean_3h,
                                telemetry_sd_3h.iloc[:, 2:6],
                                telemetry_mean_24h.iloc[:, 2:6],
                                telemetry_sd_24h.iloc[:, 2:6]], axis=1).dropna()
    
    return(telemetry_feat)


def get_lagfeatures_error(errors, telemetry):
    error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
    error_count
    error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
    print(error_count.shape)
    error_count.head(14)
    
    # combine errors for a given machine in a given hour
    error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
#     print(error_count.shape)
#     error_count.head(13)
    
    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
#     print(error_count)
#     error_count.describe()
    
    temp = []
    fields = ['error%d' % i for i in range(1,6)]
    for col in fields:
        temp.append(pd.DataFrame.rolling(pd.pivot_table(error_count,
                                                   index='datetime',
                                                   columns='machineID',
                                                   values=col), window=24).sum().resample('3H',
                                                                                 closed='left',
                                                                                 label='right').first().unstack())
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace=True)
    error_count = error_count.dropna()
#     print(error_count)
    error_count.describe()
    
    return(error_count)
    
    
def component_rep(telemetry, maint):
    

    # create a column for each error type
    comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

    # combine repairs for a given machine in a given hour
    comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

    # add timepoints where no components were replaced
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                          on=['datetime', 'machineID'],
                                                          how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

    components = ['comp1', 'comp2', 'comp3', 'comp4']
    df = comp_rep

    for comp in components:
        # convert indicator to most recent date of component change
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        #print("1", comp_rep)

        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
        #print("2",comp_rep)

        # forward-fill the most-recent date of component change
        comp_rep[comp] = comp_rep[comp].fillna(method='ffill')
        #print("3",comp_rep)


    # remove dates in 2014 (may have NaN or future component change dates)    
    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]
    for comp in components:

        comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp].values.astype('datetime64[ns]')) / np.timedelta64(1, 'D')

    df = comp_rep

    return(comp_rep)

def get_final_features(telemetry_feat, error_count, comp_rep, machines):
    
    final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on=['machineID'], how='left')
    return(final_feat)
    
def label_construct(final_feat, failures):

    
    labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
    labeled_features['model'] = labeled_features['model'].astype('object')
    labeled_features['failure'] = labeled_features['failure'].astype('object')
    labeled_features = labeled_features.fillna(method='bfill', axis=0, limit=7) # fill backward up to 24h[(7+1)*3]
    labeled_features = labeled_features.fillna('none')
    fail_data = labeled_features['failure']
    labeled_features['failure'] = fail_data.replace(['comp1', 'comp2', 'comp3', 'comp4', 'none'],[0,1,2,3,4])
    return(labeled_features)

    
    
    
    

    
    

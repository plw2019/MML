import pandas as pd
import numpy as np
import random
import os


# to ensure reproduction
def setup_seed(seed):
    np.random.seed(seed) # initialize numpy by seed
    random.seed(seed) # initialize random library by seed
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(2025)

# domain_names_map = {'book':1, 'music':2, 'movie':3}
domain_names_map = {'Books':1,'CDs_and_Vinyl':2,'Movies_and_TV':3}
# the reason might be the length of interaction is less than 2
# there is no interaction in training dataset
save_prefix = r'Amazon_platform/'
read_prefix = r'processed_data_all'

def data_partition(fname, save_prefix=save_prefix, read_prefix=read_prefix):
    # read dataset
    # names = ['user_id', 'item_id', 'timestamp', 'domain_id']
    # types = {'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32, 'domain_id': np.int32}
    names = ['user_id', 'item_id', 'timestamp']
    types = {'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32}

    train_dataset = pd.read_csv(r'{0:}/{1:}_train.csv'.format(read_prefix, fname), names=names, header=None,
                                dtype=types)
    valid_dataset = pd.read_csv(r'{0:}/{1:}_valid.csv'.format(read_prefix, fname), names=names, header=None,
                                dtype=types)
    test_dataset = pd.read_csv(r'{0:}/{1:}_test.csv'.format(read_prefix, fname), names=names, header=None,
                               dtype=types)
    neg_dataset = pd.read_csv(r'{0:}/{1:}_negative.csv'.format(read_prefix, fname), header=None)

    # sort by timestamp
    dfx = pd.DataFrame()
    dfx = pd.concat([dfx, train_dataset], axis=0)
    dfx = pd.concat([dfx, valid_dataset], axis=0)
    dfx = pd.concat([dfx, test_dataset], axis=0)
    dfx['domain_id'] = domain_names_map[fname]
    dfx.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    dfx.reset_index(drop=True, inplace=True)

    print(dfx.shape)

    df_test = dfx.groupby(['user_id']).tail(1)
    print(df_test.shape)
    dfx.drop(df_test.index, axis='index', inplace=True)
    print(dfx.shape)
    df_valid = dfx.groupby(['user_id']).tail(1)
    print(df_valid.shape)
    dfx.drop(df_valid.index, axis='index', inplace=True)
    print(dfx.shape)

    # create dictionary for saving
    # output data file
    if not os.path.exists('{0:}/{1:}'.format(save_prefix, fname)):
        os.makedirs('{0:}/{1:}'.format(save_prefix, fname))

    df_valid.to_csv(r'{0:}/{1:}/valid.csv'.format(save_prefix, fname), header=False, index=False)
    df_test.to_csv(r'{0:}/{1:}/test.csv'.format(save_prefix, fname), header=False, index=False)
    dfx.to_csv(r'{0:}/{1:}/train.csv'.format(save_prefix, fname), header=False, index=False)
    neg_dataset.to_csv(r'{0:}/{1:}/negative.csv'.format(save_prefix, fname), header=False, index=False)

# process each domain
for domain in domain_names_map.keys():
    print("Processing {0:} dataset...".format(domain))
    data_partition(domain)
# The split method
# @time:2023/10/23
# @function: if the user's interacted sequence equals 1, the model can be trained
# @time:2023/10/26
# @function: the split method is adopted
import random

import pandas as pd
import numpy as np
import json
from multiprocessing import Process, Queue
import copy


from tqdm import tqdm
import os

# set map domain_name into domain_id and vice versa
# domain_map = {1: "Books", 2: "CDs_and_Vinyl", 3: "Movies_and_TV"}
# reversed_domain_map = {"Books": 1, "CDs_and_Vinyl": 2, "Movies_and_TV": 3}
domain_map = {}
reversed_domain_map = {}


# sampler for batch generation
# get an item in item_set but not in ts
def random_neq(item_set, ts):
    t = np.random.choice(item_set)
    while t in ts:
        t = np.random.choice(item_set)
    return t


def popular_sampling(data, sample_num):
    """
    Popular sampling
    """
    item_count = data['item_id'].value_counts()
    item_count = item_count.sort_values(ascending=False)
    # print(item_count)
    data = item_count.index.values[:sample_num]
    return data


# get negative sequence, positive sequence and position sequence for each user
def sample_function(domain_invariant_user_train, user_set_in_all_domains, item_sets, argumentation_methods, batch_size, maxlen, result_queue,
                    SEED, domain_popular_items=None, args = None):

    mask_id = 1
    for domain_id in domain_map:
        mask_id = np.max([mask_id, np.max(item_sets[domain_map[domain_id]])])
    mask_id = mask_id + 1
    def sample():
        user = np.random.choice(user_set_in_all_domains)

        # the user doesn't appear in the training dataset
        while user not in domain_invariant_user_train.keys() or len(domain_invariant_user_train[user]) <= 1:
            user = np.random.choice(user_set_in_all_domains)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        seq_domain_switch_flag = np.zeros([maxlen], dtype=np.bool_)
        # save the positive sequence and the next predicted sequence respectively
        seq_dom = np.zeros([maxlen], dtype=np.int32)
        pos_dom = np.zeros([maxlen], dtype=np.int32)

        nxt = domain_invariant_user_train[user]['seq'][-1]
        nxt_dom = domain_invariant_user_train[user]['domain'][-1]
        idx = maxlen - 1

        # [...,last but one]
        domain_len = {} # ensure the maxlen is same in each domain
        for domain_id in domain_map:
            domain_len[domain_id] = int(maxlen / len(domain_map.keys()))
        # break, if each domain reach the max length
        # print(domain_len)

        # get the seq in each domain
        common_domain_len = int(maxlen / len(domain_map.keys()))


        # get the seq in each domain
        seq_single_domain = np.zeros([len(reversed_domain_map.keys()), common_domain_len], dtype=np.int32)
        pos_cur_domain = np.zeros([len(reversed_domain_map.keys()), common_domain_len], dtype=np.int32)
        neg_cur_domain = np.zeros([len(reversed_domain_map.keys()), common_domain_len], dtype=np.int32)
        pos_other_domain = np.zeros([len(reversed_domain_map.keys()), common_domain_len], dtype=np.int32)
        neg_other_domain = np.zeros([len(reversed_domain_map.keys()), common_domain_len], dtype=np.int32)
        lg_dom = np.zeros([len(reversed_domain_map.keys()), common_domain_len], dtype=np.int32) # save domain id for next interacted item in other domain


        pos_single_domain_compress = np.zeros([maxlen], dtype=np.int32)
        neg_single_domain_compress = np.zeros([maxlen], dtype=np.int32)

        nxt_single_domain_item = {}  # the next predicted item in this domain
        nxt_single_domain_id = {}  # record the position of next predicted item
        cur_single_domain = {}  # the current trained item in this domain


        nxt_single_domain_item[nxt_dom] = domain_invariant_user_train[user]['seq'][-1]
        nxt_single_domain_id[nxt_dom] = maxlen - 1

        domain_len[nxt_dom] -= 1

        # construct mapping matrix
        cur_domain_posn = {}
        for domain_id in domain_map:
            cur_domain_posn[domain_id] = common_domain_len - 1

        # indices for transformation

        domain_switch_behavior = np.zeros([len(reversed_domain_map.keys()), maxlen],
                                          dtype=np.int32)  # the nest item, the domain is same as the output
        next_behavior = np.zeros([len(reversed_domain_map.keys()), maxlen],
                                 dtype=np.int32)  # the domain is different as the output
        domain_switch_behavior_id = {}  # record the index of the last item id in domain_switch_behavior
        next_behavior_id = {}  # record the index of the last item id in next behavior

        reconstructed_user_interacted_seq = [] # are leveraged for data argumentation
        reconstructed_user_interacted_domain = []
        domain_count = {} # record the numter of the interaction
        for domain_id in domain_map:
            domain_count[domain_id] = 0

        each_domain_seq = {}
        for domain_id in domain_map:
            each_domain_seq[domain_id] =[]

        ts = set(domain_invariant_user_train[user]['seq'])
        for item_id, domain_id in zip(reversed(domain_invariant_user_train[user]['seq'][:-1]),
                                      reversed(domain_invariant_user_train[user]['domain'][:-1])):
            # judge the current item is jumped or not
            if domain_len[domain_id] <= 0:
                continue

            domain_len[domain_id] -= 1
            seq[idx] = item_id
            seq_dom[idx] = domain_id
            each_domain_seq[domain_id].append(item_id)

            domain_count[domain_id] += 1

            reconstructed_user_interacted_seq.append(item_id)
            reconstructed_user_interacted_domain.append(domain_id)

            seq_domain_switch_flag[idx] = (domain_id != nxt_dom)
            if seq_domain_switch_flag[idx]:
                if nxt_dom not in next_behavior_id:
                    try:
                        next_behavior_id[nxt_dom] = maxlen - 1
                        next_behavior[nxt_dom - 1, next_behavior_id[nxt_dom]] = item_id
                    except:
                        print(nxt_dom)
                else:
                    try:
                        next_behavior_id[nxt_dom] -= 1
                        next_behavior[nxt_dom - 1, next_behavior_id[nxt_dom]] = item_id
                    except:
                        print(nxt_dom)

                if domain_id not in domain_switch_behavior_id and domain_id in next_behavior_id:
                    domain_switch_behavior_id[domain_id] = maxlen - 1
                    domain_switch_behavior[domain_id - 1, domain_switch_behavior_id[domain_id]] = item_id
                elif domain_id in domain_switch_behavior_id and domain_id in next_behavior_id:
                    domain_switch_behavior_id[domain_id] -= 1
                    domain_switch_behavior[domain_id - 1, domain_switch_behavior_id[domain_id]] = item_id

            if nxt != 0: neg[idx] = random_neq(item_sets[domain_map[nxt_dom]], ts)

            if domain_id not in nxt_single_domain_id.keys():
                neg_item = random_neq(item_sets[domain_map[domain_id]], ts)
                nxt_single_domain_id[domain_id] = idx
                nxt_single_domain_item[domain_id] = item_id
                seq_single_domain[domain_id - 1, cur_domain_posn[domain_id]] = item_id
                if domain_id != nxt_dom:
                    pos_other_domain[domain_id - 1, cur_domain_posn[domain_id]] = nxt
                    lg_dom[domain_id - 1, cur_domain_posn[domain_id]] = nxt_dom
                    neg_other_domain[domain_id - 1, cur_domain_posn[domain_id]] = neg_item
                neg_cur_domain[domain_id - 1, cur_domain_posn[domain_id]] = neg_item
                cur_domain_posn[domain_id] -= 1
            else:
                neg_item = random_neq(item_sets[domain_map[domain_id]], ts)
                seq_single_domain[domain_id - 1, cur_domain_posn[domain_id]] = item_id
                pos_cur_domain[domain_id - 1, cur_domain_posn[domain_id]] = nxt_single_domain_item[domain_id]
                if domain_id != nxt_dom:
                    pos_other_domain[domain_id - 1, cur_domain_posn[domain_id]] = nxt
                    lg_dom[domain_id - 1, cur_domain_posn[domain_id]] = nxt_dom
                    neg_other_domain[domain_id - 1, cur_domain_posn[domain_id]] = neg_item  # useless, only alignment
                neg_cur_domain[domain_id - 1, cur_domain_posn[domain_id]] = neg_item  # useless, only alignment
                # update next item and position
                cur_domain_posn[domain_id] -= 1
                nxt_single_domain_id[domain_id] = idx  # useless, only alignment
                nxt_single_domain_item[domain_id] = item_id

            pos_dom[idx] = nxt_dom
            pos[idx] = nxt

            nxt = item_id
            nxt_dom = domain_id
            idx -= 1
            if idx == -1: break

        for domain_id in domain_map:
            if domain_id in next_behavior_id.keys() and domain_id in domain_switch_behavior_id.keys():
                if next_behavior_id[domain_id] < domain_switch_behavior_id[domain_id]:
                    next_behavior[domain_id - 1, next_behavior_id[domain_id]] = 0  # more than one
            if domain_id in next_behavior_id.keys() and domain_id not in domain_switch_behavior_id.keys():
                next_behavior[domain_id - 1, next_behavior_id[domain_id]] = 0  # more than one




        # data argumentation
        arg_seq = reconstructed_user_interacted_seq.copy()
        mask_item = mask_id # mask item is mask id

        pos_single_other_domain_compress = pos * seq_domain_switch_flag  # the judge of the domain switch

        #global argumentation_ratio
        argumentation_ratio = args.argumentation_ratio



        #print(argumentation_methods)
        rand_method = random.choice(argumentation_methods)
        # 'resort','mask','data_balance'

        if rand_method == 'resort':
            #resort


            len_arg_seq = len(arg_seq)

            arg_seq = np.asarray(arg_seq, dtype=np.int32)
            arg_seq = np.flip(arg_seq) # get the reversed interacted sequence in the last step

            start = np.random.randint(0, len_arg_seq)
            if start < 0:
                start = 0

            np.random.shuffle(arg_seq[start: int(len_arg_seq * argumentation_ratio + start + 1)])

            if len_arg_seq >= maxlen:
                arg_seq = arg_seq[-maxlen:]
            else:
                padding = np.asarray([0] * (maxlen - len_arg_seq), dtype=np.int32)
                arg_seq = np.concatenate((padding, arg_seq), axis=0)
        elif rand_method == 'mask':
            #mask

            len_arg_seq = len(arg_seq)

            arg_seq = np.asarray(arg_seq, dtype=np.int32)
            arg_seq = np.flip(arg_seq) # get the reversed interacted sequence in the last step

            to_be_masked = np.random.randint(0, len_arg_seq, size=int(len_arg_seq * argumentation_ratio))
            # Return random integers from low (inclusive) to high (exclusive).

            arg_seq[to_be_masked] = mask_item
            #print(mask_item)

            if len_arg_seq >= maxlen:
                arg_seq = arg_seq[-maxlen:]
            else:
                padding = np.asarray([0] * (maxlen - len_arg_seq), dtype=np.int32)
                arg_seq = np.concatenate((padding, arg_seq), axis=0)
        elif rand_method == 'crop':
            #crop
            len_arg_seq = len(arg_seq)

            arg_seq = np.asarray(arg_seq, dtype=np.int32)
            arg_seq = np.flip(arg_seq)  # get the reversed interacted sequence in the last step

            start = np.random.randint(0, len_arg_seq)
            if start < 0:
                start = 0
            crop_len = int(len_arg_seq * argumentation_ratio) + 1
            # arg_seq = np.delete(arg_seq, slice(start,start + crop_len))
            arg_seq = arg_seq[start: start + crop_len]
            len_arg_seq = len(arg_seq) # update the length of the sequence for padding items
            if len_arg_seq >= maxlen:
                arg_seq = arg_seq[-maxlen:]
            else:
                padding = np.asarray([0] * (maxlen - len_arg_seq), dtype=np.int32)
                arg_seq = np.concatenate((padding, arg_seq), axis=0)
        elif rand_method == 'data_balance':
            sorted_data = sorted(domain_count.items(), key=lambda x: x[1])

            # get the interacted min domain
            min_domain_id = sorted_data[0][0]
            arg_seq = np.zeros([maxlen], dtype=np.int32)
            arg_idx = maxlen - 1

            min_domain_len = {} # record the numter of the interaction
            for domain_id in domain_map:
                min_domain_len[domain_id] = domain_count[min_domain_id]

            for item_id, domain_id in zip(reconstructed_user_interacted_seq,reconstructed_user_interacted_domain):
                if min_domain_len[domain_id] <= 0:
                    arg_idx -= 1
                    continue
                min_domain_len[domain_id] -= 1

                arg_seq[arg_idx] = item_id
                arg_idx -= 1







        # print(seq)
        # print(seq_dom)
        # print(pos)
        # print(pos_dom)
        # print(seq_single_domain)
        # print(pos_cur_domain)
        # print(neg_cur_domain)
        # print(pos_other_domain)
        # print(neg_other_domain)
        # print(lg_dom)
        # exit(0)

        return (user, seq, pos, neg, seq_dom, pos_dom, seq_single_domain, pos_cur_domain, neg_cur_domain, domain_switch_behavior, next_behavior, arg_seq, pos_other_domain, neg_other_domain, lg_dom)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, domain_invariant_user_train, user_set_in_all_domains, item_sets, argumentation_methods, batch_size=64, maxlen=10,
                 n_workers=1, domain_popular_items=None, args = None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(domain_invariant_user_train,
                                                      user_set_in_all_domains,
                                                      item_sets,
                                                      argumentation_methods,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      domain_popular_items,
                                                      args
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def convert_json_key(param_dict):
    """
    json.dump不支持key是int的dict，在编码存储的时候会把所有的int型key写成str类型的
    所以在读取json文件后，用本方法将所有的被解码成str的int型key还原成int
    """
    new_dict = dict()
    for key, value in param_dict.items():
        if isinstance(value, (dict,)):
            res_dict = convert_json_key(value)
            try:
                new_key = int(key)
                new_dict[new_key] = res_dict
            except:
                new_dict[key] = res_dict
        else:
            try:
                new_key = int(key)
                new_dict[new_key] = value
            except:
                new_dict[key] = value

    return new_dict


# train/val/test data generation
def data_partition_save(datasets_information, args):
    # define the columns of the datasets
    names = ['user_id', 'item_id', 'timestamp', 'domain']
    types = {'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32, 'domain': np.int32}
    # read training dataset, validation dataset and test dataset in all domains
    # keys: [domain] values: data
    all_train_datasets = {}
    all_validation_datasets = {}
    all_test_datasets = {}
    all_negative_datasets = {}
    data_path = datasets_information['data_path']
    for domain in datasets_information['domains']:
        all_train_datasets[domain] = pd.read_csv(r'{0:}/{1:}/train.csv'.format(data_path, domain), names=names,
                                                 header=None, dtype=types)
        # sort by timestamp
        all_train_datasets[domain] = all_train_datasets[domain].sort_values(by=['timestamp'], ascending=True)
        all_validation_datasets[domain] = pd.read_csv(r'{0:}/{1:}/valid.csv'.format(data_path, domain), names=names,
                                                      header=None, dtype=types)
        all_test_datasets[domain] = pd.read_csv(r'{0:}/{1:}/test.csv'.format(data_path, domain), names=names,
                                                header=None, dtype=types)
        all_negative_datasets[domain] = pd.read_csv(r'{0:}/{1:}/negative.csv'.format(data_path, domain), header=None)
    # get item sets and user sets in all domains respectively
    # one domain has one item set and one user set
    item_sets = {}
    user_sets = {}
    for domain in datasets_information['domains']:
        dfx = pd.DataFrame()
        dfx = pd.concat([dfx, all_train_datasets[domain]], axis=0)
        dfx = pd.concat([dfx, all_validation_datasets[domain]], axis=0)
        dfx = pd.concat([dfx, all_test_datasets[domain]], axis=0)
        item_sets[domain] = dfx.item_id.unique()
        user_sets[domain] = dfx.user_id.unique()
    # get max item_id and max user_id in all domains
    max_item_id = 0
    max_user_id = 0
    for domain in datasets_information['domains']:
        max_item_id = max(max_item_id, max(item_sets[domain]))
        max_user_id = max(max_user_id, max(user_sets[domain]))
    # get the unified training dataset in all domains
    unified_training_dataset = pd.DataFrame()
    for domain in datasets_information['domains']:
        unified_training_dataset = pd.concat([unified_training_dataset, all_train_datasets[domain]], axis=0)
    unified_training_dataset = unified_training_dataset.sort_values(by=['timestamp'],
                                                                    ascending=True)  # sort by timestamp
    # get item set and user set in all domains
    # all domains have only one item set and one user set
    # the int32 can't be saved
    # unified_training_dataset.user_id.unique()

    # validation/test [domain][user][seq/target/domain/target_timestamp/seq_timestamp]
    # the position is considered later
    # neg_cand [domain][user]

    # training [user][domain][seq/position/timestamp] domain-specific
    # position denotes the position
    # training [user][seq/domain/timestamp] domain-invariant
    # get training datasets for domain-invariant preference and domain-specific preference respectively.
    domain_specific_user_train = {}  # In each domain, the user has one interaction sequence.
    domain_invariant_user_train = {}  # the user has only one interaction sequence in all domains
    # the users in the training dataset contains all users
    user_set_in_all_domains = unified_training_dataset.user_id.unique()
    # get training datasets for domain-invariant preference
    for user in user_set_in_all_domains:
        domain_invariant_user_train[int(user)] = {}  # create empty dictionary
        domain_invariant_user_train[int(user)]['seq'] = unified_training_dataset[unified_training_dataset['user_id']
                                                                                 == user]['item_id'].values.tolist()
        domain_invariant_user_train[int(user)]['domain'] = unified_training_dataset[unified_training_dataset['user_id']
                                                                                    == user]['domain'].values.tolist()
        domain_invariant_user_train[int(user)]['timestamp'] = \
        unified_training_dataset[unified_training_dataset['user_id']
                                 == user]['timestamp'].values.tolist()
    # the position_id of the item that user interacted with in the last time in training dataset is maxlen - 1
    # record the position of items that the users interacted with
    maxlen = args.maxlen  # the max length of the sequence
    # get training datasets for domain-specific preference
    # TypeError: keys must be str, int, float, bool or None, not int32
    # TypeError: Object of type int32 is not JSON serializable
    for user in user_set_in_all_domains:
        # set empty dictionary
        domain_specific_user_train[int(user)] = {}
        for domain in datasets_information['domains']:
            domain_specific_user_train[int(user)][domain] = {}
        for domain in datasets_information['domains']:
            domain_specific_user_train[int(user)][domain]['seq'] = all_train_datasets[domain][
                all_train_datasets[domain]['user_id'] == user]['item_id'].values.tolist()
            domain_specific_user_train[int(user)][domain]['timestamp'] = all_train_datasets[domain][
                all_train_datasets[domain]['user_id'] == user]['timestamp'].tolist()
    # get negative candidates for each user in all domains
    neg_cand = {}
    for domain in datasets_information['domains']:
        neg_cand[domain] = {}
        for user in user_set_in_all_domains:
            neg_cand[domain][int(user)] = \
            all_negative_datasets[domain][all_negative_datasets[domain][0] == user].values[0][1:].tolist()
    # combine training dataset and validation dataset for testing
    unified_training_and_validation_dataset = pd.DataFrame()
    unified_training_and_validation_dataset = pd.concat([unified_training_and_validation_dataset,
                                                         unified_training_dataset], axis=0)
    for domain in datasets_information['domains']:
        unified_training_and_validation_dataset = pd.concat([unified_training_and_validation_dataset,
                                                             all_validation_datasets[domain]], axis=0)
    unified_training_and_validation_dataset = unified_training_and_validation_dataset.sort_values(by=['timestamp'],
                                                                                                  ascending=True, kind='mergesort')

    # combine training dataset, validation dataset and test dataset for getting the set
    unified_training_validation_and_test_dataset = pd.DataFrame()
    unified_training_validation_and_test_dataset = pd.concat([unified_training_validation_and_test_dataset,
                                                              unified_training_and_validation_dataset], axis=0)
    for domain in datasets_information['domains']:
        unified_training_validation_and_test_dataset = pd.concat([unified_training_validation_and_test_dataset,
                                                                  all_test_datasets[domain]], axis=0)
    unified_training_validation_and_test_dataset = unified_training_validation_and_test_dataset.sort_values(by=['timestamp'],
                                                                                                  ascending=True, kind='mergesort')


    # item set should contain the items in the training dataset, the validation dataset and the test dataset
    item_set_in_all_domains = unified_training_validation_and_test_dataset.item_id.unique()
    user_set_in_all_domains = unified_training_validation_and_test_dataset.user_id.unique()

    # get validation data and test data for each user
    user_test = {}
    user_valid = {}
    for domain in datasets_information['domains']:
        # create empty dictionary
        user_test[domain] = {}
        user_valid[domain] = {}
        for user in user_set_in_all_domains:
            # the data framework for this user
            df_test = all_test_datasets[domain][all_test_datasets[domain]['user_id'] == user]
            df_valid = all_validation_datasets[domain][all_validation_datasets[domain]['user_id'] == user]

            # get sequence, target and domain that user interacted with for each user
            if df_test.shape[0] != 0:
                # create empty dictionary, not empty
                user_test[domain][int(user)] = {}
                # get historical items for current user
                df_for_user = unified_training_and_validation_dataset[
                    unified_training_and_validation_dataset['user_id'] == user]
                df_for_user = df_for_user[df_for_user['timestamp'] <= df_test.timestamp.values[
                    0]]  # The df_for_user doesn't contain the test dataset
                user_test[domain][int(user)]['seq'] = df_for_user.item_id.values.tolist()
                user_test[domain][int(user)]['target'] = int(df_test.item_id.values[0])
                user_test[domain][int(user)]['domain'] = df_for_user.domain.values.tolist()
                user_test[domain][int(user)]['seq_timestamp'] = df_for_user.timestamp.values.tolist()
                user_test[domain][int(user)]['target_timestamp'] = int(df_test.timestamp.values[0])

            if df_valid.shape[0] != 0:
                # create empty dictionary
                user_valid[domain][int(user)] = {}
                # get historical items for current user
                df_for_user = unified_training_dataset[
                    unified_training_dataset['user_id'] == user]
                df_for_user = df_for_user[df_for_user['timestamp'] <= df_valid.timestamp.values[0]]
                user_valid[domain][int(user)]['seq'] = df_for_user.item_id.values.tolist()
                user_valid[domain][int(user)]['target'] = int(df_valid.item_id.values[0])
                user_valid[domain][int(user)]['domain'] = df_for_user.domain.values.tolist()
                user_valid[domain][int(user)]['seq_timestamp'] = df_for_user.timestamp.values.tolist()
                user_valid[domain][int(user)]['target_timestamp'] = int(df_valid.timestamp.values[0])

    # save the data including domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test
    # the saved path
    saved_path = r'{0:}/all'.format(data_path)
    if not os.path.isdir(saved_path):
        os.makedirs(saved_path)

    popular_sampling_domains = {}
    # popular sampling
    for domain in datasets_information['domains']:
        item_set = set(all_train_datasets[domain]['item_id'])
        sample_num = int(len(item_set) * 0.01)
        popular_items = popular_sampling(all_train_datasets[domain], sample_num)
        popular_sampling_domains[domain] = popular_items.tolist()

    # {dict} 多加了不该加的括号
    with open(os.path.join(saved_path, 'domain_specific_user_train.json'), 'w') as f:
        json.dump(domain_specific_user_train, f)
    with open(os.path.join(saved_path, 'domain_invariant_user_train.json'), 'w') as f:
        json.dump(domain_invariant_user_train, f)
    with open(os.path.join(saved_path, 'neg_cand.json'), 'w') as f:
        json.dump(neg_cand, f)
    with open(os.path.join(saved_path, 'user_valid.json'), 'w') as f:
        json.dump(user_valid, f)
    with open(os.path.join(saved_path, 'user_test.json'), 'w') as f:
        json.dump(user_test, f)
    with open(os.path.join(saved_path, 'domain_popular_items.json'), 'w') as f:
        json.dump(popular_sampling_domains, f)

    return [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
            domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test, popular_sampling_domains]


def data_partition_load(datasets_information, args):
    # define the columns of the datasets
    names = ['user_id', 'item_id', 'timestamp', 'domain']
    types = {'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32, 'domain': np.int32}
    # read training dataset, validation dataset and test dataset in all domains
    # keys: [domain] values: data
    all_train_datasets = {}
    all_validation_datasets = {}
    all_test_datasets = {}
    all_negative_datasets = {}
    data_path = datasets_information['data_path']
    for domain in datasets_information['domains']:
        all_train_datasets[domain] = pd.read_csv(r'{0:}/{1:}/train.csv'.format(data_path, domain), names=names,
                                                 header=None, dtype=types)
        # sort by timestamp
        all_train_datasets[domain] = all_train_datasets[domain].sort_values(by=['timestamp'], ascending=True)
        all_validation_datasets[domain] = pd.read_csv(r'{0:}/{1:}/valid.csv'.format(data_path, domain), names=names,
                                                      header=None, dtype=types)
        all_test_datasets[domain] = pd.read_csv(r'{0:}/{1:}/test.csv'.format(data_path, domain), names=names,
                                                header=None, dtype=types)
        all_negative_datasets[domain] = pd.read_csv(r'{0:}/{1:}/negative.csv'.format(data_path, domain), header=None)
    # get item sets and user sets in all domains respectively
    # one domain has one item set and one user set
    item_sets = {}
    user_sets = {}
    for domain in datasets_information['domains']:
        dfx = pd.DataFrame()
        dfx = pd.concat([dfx, all_train_datasets[domain]], axis=0)
        dfx = pd.concat([dfx, all_validation_datasets[domain]], axis=0)
        dfx = pd.concat([dfx, all_test_datasets[domain]], axis=0)
        item_sets[domain] = dfx.item_id.unique()
        user_sets[domain] = dfx.user_id.unique()
    # get max item_id and max user_id in all domains
    max_item_id = 0
    max_user_id = 0
    for domain in datasets_information['domains']:
        max_item_id = max(max_item_id, max(item_sets[domain]))
        max_user_id = max(max_user_id, max(user_sets[domain]))
    # get the unified training dataset in all domains
    unified_training_dataset = pd.DataFrame()
    for domain in datasets_information['domains']:
        unified_training_dataset = pd.concat([unified_training_dataset, all_train_datasets[domain]], axis=0)
    unified_training_dataset = unified_training_dataset.sort_values(by=['timestamp'],
                                                                    ascending=True)  # sort by timestamp
    # get item set and user set in the training dataset
    # all domains have only one item set and one user set
    item_set_in_all_domains = unified_training_dataset.item_id.unique()
    user_set_in_all_domains = unified_training_dataset.user_id.unique()

    # get the item set and the user set in all domains
    for domain in datasets_information['domains']:
        item_set_in_all_domains = np.hstack([item_set_in_all_domains, item_sets[domain]])
        user_set_in_all_domains = np.hstack([user_set_in_all_domains, user_sets[domain]])

    # get the item set and the user set in the training dataset, the validation dataset and the test dataset
    item_set_in_all_domains = np.unique(item_set_in_all_domains)
    user_set_in_all_domains = np.unique(user_set_in_all_domains)

    # load data including domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test
    # the loaded path
    loaded_path = r'{0:}/all'.format(data_path)
    domain_specific_user_train = convert_json_key(json.load(
        open(r'{0:}/domain_specific_user_train.json'.format(loaded_path), 'r', encoding='utf-8')))
    domain_invariant_user_train = convert_json_key(json.load(
        open(r'{0:}/domain_invariant_user_train.json'.format(loaded_path), 'r', encoding='utf-8')))
    neg_cand = convert_json_key(json.load(
        open(r'{0:}/neg_cand.json'.format(loaded_path), 'r', encoding='utf-8')))
    user_valid = convert_json_key(json.load(
        open(r'{0:}/user_valid.json'.format(loaded_path), 'r', encoding='utf-8')))
    user_test = convert_json_key(json.load(
        open(r'{0:}/user_test.json'.format(loaded_path), 'r', encoding='utf-8')))
    domain_popular_items = convert_json_key(json.load(
        open(r'{0:}/domain_popular_items.json'.format(loaded_path), 'r', encoding='utf-8')))



    return [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
            domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test, domain_popular_items]


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
     domain_specific_user_train,
     domain_invariant_user_train, neg_cand, user_valid, user_test, domain_popular_items] = copy.deepcopy(dataset)

    # create dictionary for saving results
    ans_list = {}

    for domain in domain_map.values():
        # initialize parameters
        NDCG = 0.0
        valid_user = 0.0
        HT = 0.0
        MRR = 0.0

        # validation
        for u in tqdm(user_sets[domain], ncols=80):
            if not (u in user_valid[domain].keys() and len(user_valid[domain][u]['seq']) >= 1): continue

            # get the seq in each domain
            common_domain_len = int(args.maxlen / len(domain_map.keys()))  # the common length is same in different domains
            seq = np.zeros([common_domain_len], dtype=np.int32)
            hybrid_seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = common_domain_len - 1
            end_idx = args.maxlen - 1


            domain_len = {}  # ensure the maxlen is same in each domain
            for domain_id in domain_map:
                domain_len[domain_id] = int(args.maxlen / len(domain_map.keys()))

            for item_id, domain_id in zip(reversed(user_valid[domain][u]['seq']),
                                          reversed(user_valid[domain][u]['domain'])):
                # judge the current item is jumped or not
                if domain_len[domain_id] <= 0:
                    #print('*************')
                    continue
                domain_len[domain_id] -= 1
                # get the interaction sequence in this domain
                if domain_id == reversed_domain_map[domain]:
                    seq[idx] = item_id
                    idx -= 1


                # seq[idx] = item_id
                hybrid_seq[end_idx] = item_id

                end_idx -= 1
                # idx -= 1
                if end_idx == -1 or idx == -1: break
            # print(user_valid[domain][u]['domain'])
            # print(user_valid[domain][u]['seq'])
            # print(seq)

            item_idx = [user_valid[domain][u]['target']]
            item_idx = item_idx + neg_cand[domain][u]


            predictions = -model.predict(*[np.array(l) for l in
                                           [reversed_domain_map[domain], [seq], [hybrid_seq], item_idx]])

            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < args.top_n:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
                MRR += 1 / (rank + 1)
            # if valid_user % 100 == 0:
            #     print('.', end="")
            #     sys.stdout.flush()

        # save results
        ans_list[domain] = [NDCG / valid_user, HT / valid_user, MRR / valid_user]

    return ans_list


# evaluate on test set
def evaluate(model, dataset, args):
    [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
     domain_specific_user_train,
     domain_invariant_user_train, neg_cand, user_valid, user_test, domain_popular_items] = copy.deepcopy(dataset)

    # create dictionary for saving results
    ans_list = {}

    for domain in domain_map.values():
        # initialize parameters
        NDCG = 0.0
        HT = 0.0
        MRR = 0.0
        valid_user = 0.0

        for u in tqdm(user_sets[domain], ncols=80):
            # user interaction length is short
            if not (u in user_test[domain].keys() and len(user_test[domain][u]['seq']) >= 1): continue

            # get the seq in each domain
            common_domain_len = int(args.maxlen / len(domain_map.keys()))  # the common length is same in different domains
            seq = np.zeros([common_domain_len], dtype=np.int32)
            hybrid_seq = np.zeros([args.maxlen], dtype=np.int32)


            idx = common_domain_len - 1
            end_idx = args.maxlen - 1

            domain_len = {}  # ensure the maxlen is same in each domain
            for domain_id in domain_map:
                domain_len[domain_id] = int(args.maxlen / len(domain_map.keys()))

            # the test data contains the interaction records in validation dataset
            for item_id, domain_id in zip(reversed(user_test[domain][u]['seq']),
                                          reversed(user_test[domain][u]['domain'])):
                # judge the current item is jumped or not
                if domain_len[domain_id] <= 0:
                    #print('====================')
                    continue
                domain_len[domain_id] -= 1
                if domain_id == reversed_domain_map[domain]:
                    seq[idx] = item_id
                    idx -= 1


                # seq[idx] = item_id
                hybrid_seq[end_idx] = item_id

                end_idx -= 1
                # idx -= 1
                if end_idx == -1 or idx == -1: break

            item_idx = [user_test[domain][u]['target']]
            item_idx = item_idx + neg_cand[domain][u]

            # for _ in range(100):
            #     t = np.random.choice(item_set)
            #     while t in rated: t = np.random.choice(item_set)
            #     item_idx.append(t)`

            predictions = -model.predict(*[np.array(l) for l in
                                           [reversed_domain_map[domain], [seq], [hybrid_seq], item_idx]])
            predictions = predictions[0]  # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < args.top_n:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
                MRR += 1 / (rank + 1)
        # save results
        # append[[]]
        ans_list[domain] = [NDCG / valid_user, HT / valid_user, MRR / valid_user]

    return ans_list

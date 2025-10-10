# -*- coding: utf-8 -*-


import os
import time

import numpy as np
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm

from model import MDR_local_global
from utils import *
import random


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_information', default='config_3.json', type=str)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_domain_shared_blocks', default=1, type=int)
parser.add_argument('--num_domain_specific_blocks', default=1, type=int, help='the number of GRU layers in intra-domain sequential encoder in each domain')
parser.add_argument('--num_blocks_mixed_seq', default=1, type=int, help='the number of GRU layers in hybrid-domain sequential encoder')
parser.add_argument('--num_cross_attention_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--weight_decay', default=0.1, type=float)
parser.add_argument('--top_n', default=10, type = int)
parser.add_argument('--interval', default=50, type = int)
parser.add_argument('--early_stop', default=1, type = int)
parser.add_argument('--load_processed_data', required=True, type = str2bool)
parser.add_argument('--ll_loss_weight', default=0.5, required=True, type = float, help='The weight of loss in Task 2')
parser.add_argument('--gg_loss_weight', default=0.5, required=True, type = float, help='The weight of loss in Task 1')
parser.add_argument('--lg_loss_weight', default=0.5, required=True, type = float, help='The weight of loss in Task 3')
parser.add_argument('--behavior_regularizer_weight', default=0.1, type = float)
parser.add_argument('--using_mask_cross_attention', default=1, type = float) # If the item in current domain, the mask operation will be done
parser.add_argument('--saved_the_model', default=False, type = str2bool)
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--cls_weight', default=0.1, type=float)
parser.add_argument('--argumentation_ratio', default=0.2, type=float, help='The ratio of an data augmentation method')
parser.add_argument('--argumentation_methods', nargs='+', type=str, default=['resort','mask','data_balance'], help='The type of an data augmentation method')
parser.add_argument('--notes', default='MML_GRU4Rec', type=str)
parser.add_argument('--seed', default=2020, type=int)

'''
In your code, the default parameter value is False, which is a boolean value, not a string.
Therefore, the argparse library will not invoke the str2bool function to handle this default value.
'''

args = parser.parse_args()

# # to ensure reproduction
def setup_seed(seed):
    torch.manual_seed(seed) # initialize CPU,GPU by seed
    torch.cuda.manual_seed(seed) # initialize current GPU by seed
    torch.cuda.manual_seed_all(seed) # initialize all GPUs by seed
    np.random.seed(seed) # initialize numpy by seed
    random.seed(seed) # initialize random library by seed
    torch.backends.cudnn.deterministic = True # avoid the algorithm isn't the deterministic algorithm

    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = args.seed
setup_seed(SEED)


if __name__ == '__main__':
    print(SEED)
    # get the information of the dataset
    datasets_information = json.load(open(args.datasets_information, 'r', encoding='utf-8'))
    # create the map function
    '''
    domain_map = {1:"Books", 2:"CDs_and_Vinyl", 3:"Movies_and_TV"}
    reversed_domain_map = {"Books":1, "CDs_and_Vinyl":2, "Movies_and_TV":3}
    '''
    # in utils.py
    for domain_id,domain_name in enumerate(datasets_information['domains']):
        domain_map[domain_id + 1] = domain_name
        reversed_domain_map[domain_name] = domain_id + 1

    print(domain_map)
    print(reversed_domain_map)

    hyper_parameters = vars(args)
    t_save = time.gmtime()
    #notes = 'parameter_search_loss' # change the research space
    # notes = 'data_imbalance7'
    notes = args.notes
    information = time.strftime('%Y_%m_%d_%H_%M_%S', t_save) + '-' + notes
    hyper_parameters['information'] = information  # record_information

    if not os.path.isdir(os.path.join(notes, 'all')):
        os.makedirs(os.path.join(notes, 'all'))

    # make files for saving results
    for domain in datasets_information['domains']:
        if not os.path.isdir(os.path.join(notes, domain + '_' + args.train_dir)):
            os.makedirs(os.path.join(notes, domain + '_' + args.train_dir))

    # global dataset
    if not args.load_processed_data:
        dataset = data_partition_save(datasets_information, args)
    else:
        dataset = data_partition_load(datasets_information, args)

    [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
     domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test, domain_popular_items] = dataset
    num_batch = len(
        domain_invariant_user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    len_domains = {}
    # get the average training sequence length in all domains
    cc = 0.0
    for u in domain_invariant_user_train:
        cc += len(domain_invariant_user_train[u]['seq'])
    print('average training sequence length in all domains: %.2f' % (cc / len(domain_invariant_user_train)))
    len_domains['all'] = cc / len(domain_invariant_user_train)
    # get the average training sequence length in each domain
    cc = 0.0
    for domain in datasets_information['domains']:
        cc = 0.0
        count_seq = 0  # the count of the sequence
        for u in domain_specific_user_train:
            cc += len(domain_specific_user_train[u][domain]['seq'])
            count_seq += 1
        print('average training sequence length in %s : %.2f' % (domain, cc / count_seq))
        len_domains[domain] = cc / count_seq

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    sampler = WarpSampler(domain_invariant_user_train, user_set_in_all_domains, item_sets, args.argumentation_methods,
                          batch_size=args.batch_size,
                          maxlen=args.maxlen, n_workers=3, domain_popular_items=domain_popular_items, args=args)
    # The model named MML+GRU4Rec
    model = MDR_local_global(max_user_id, max_item_id, datasets_information, args).to(args.device)  # no ReLU activation in original SASRec implementation?



    loss_type = ['ll_loss','gg_loss','lg_loss','sum_loss']
    regularizer_loss = ['behavior_regularizer']


    # create dictionary for saving results
    ans_list = {}
    for domain in datasets_information['domains']:
        ans_list[domain] = []

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    '''
    parameters_in_this_domain = {
            'sequence_length_in_current_domain': len_domains[domain],
            'sequence_length_in_all_domains': len_domains['all']
        }
        parameters_domains[domain] = {
                    **hyper_parameters,**parameters_in_this_domain
        }
    '''
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        best_result_domains = {}
        parameters_domains = {}
        for domain in datasets_information['domains']:
            best_result_domains[domain] = {}
            print(
                'domain:%s, test (NDCG@%d: %.4f, HR@%d: %.4f, MRR@%d: %.4f)'
                % (domain, args.top_n, t_test[domain][0], args.top_n, t_test[domain][1],
                   args.top_n, t_test[domain][2]))
            best_result_domains[domain]['test_ndcg_10'] = t_test[domain][0]
            best_result_domains[domain]['test_hr_10'] = t_test[domain][1]
            best_result_domains[domain]['test_mrr_10'] = t_test[domain][2]

            parameters_in_this_domain = {
                'sequence_length_in_current_domain': len_domains[domain],
                'sequence_length_in_all_domains': len_domains['all']
            }
            parameters_domains[domain] = {
                **hyper_parameters, **parameters_in_this_domain
            }

        if not os.path.isdir(os.path.join(notes, 'all')):
            os.makedirs(os.path.join(notes, 'all'))
        with open(os.path.join(notes, 'all', '{0:}_parameters.json'.format(information)), 'w') as f:
            json.dump(parameters_domains, f)

        with open(os.path.join(notes, 'all', '{0:}_results.json'.format(information)), 'w') as f:
            json.dump(best_result_domains, f)
        exit(0)

        #print('test (NDCG@10: %.4f, HR@10: %.4f, MRR@10: %.4f)' % (t_test[0], t_test[1], t_test[2]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss(), reduce = mean
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    count = 0  # early_stop

    # save the results in each domain
    best_result = {}  # record best result during training
    train_loss = [] # record the loss during the training
    val_hr_10, val_ndcg_10, val_mrr_10, test_hr_10, test_ndcg_10, test_mrr_10 = {}, {}, {}, {}, {}, {}
    # initialize the best result to 0
    for domain in datasets_information['domains']:
        best_result[domain] = 0
        val_hr_10[domain], val_ndcg_10[domain], val_mrr_10[domain] = 0, 0, 0
        test_hr_10[domain], test_ndcg_10[domain], test_mrr_10[domain] = 0, 0, 0
    sum_best_performance = 0 # the best performance
    sum_performance = 0 # the performance during test


    # training, validating, testing model
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1), ncols=80):
        if args.inference_only: break  # just to decrease identition
        sum_loss = 0


        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, seq_dom, pos_dom, seq_sd, pos_cd, neg_cd, domain_switch_behavior, next_behavior, arg_seq, pos_oth_dom, neg_oth_dom, lg_dom = sampler.next_batch()  # tuples to ndarray

            u, seq, pos, neg = np.asarray(u), np.asarray(seq), np.asarray(pos), np.asarray(neg)
            seq_dom, pos_dom = np.asarray(seq_dom), np.asarray(pos_dom)
            seq_sd, pos_cd, neg_cd = np.asarray(seq_sd), np.asarray(pos_cd), np.asarray(neg_cd)
            domain_switch_behavior, next_behavior = np.asarray(domain_switch_behavior), np.asarray(next_behavior)
            arg_seq = np.asarray(arg_seq)
            pos_oth_dom = np.asarray(pos_oth_dom)
            neg_oth_dom = np.asarray(neg_oth_dom)
            lg_dom = np.asarray(lg_dom)

            ll_pos_logits, ll_neg_logits, gg_pos_logits, gg_neg_logits, lg_pos_logits, lg_neg_logits, cls_loss\
                = model(seq, seq_sd, pos_cd, neg_cd, pos, neg, arg_seq, pos_oth_dom, neg_oth_dom, lg_dom)
            optimizer.zero_grad()
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            # reshape pos_logits and neg_logits
            #pos_logits, neg_logits = pos_logits.view(seq.shape[0], -1), neg_logits.view(seq.shape[0],-1)  # {B,num_domain,L] -> [B,num_domain*L]
            #pos_all_domains = pos_sd.reshape(seq.shape[0], -1)  # {B,num_domain,L] -> [B,num_domain*L]

            ll_pos_labels, ll_neg_labels = torch.ones(ll_pos_logits[0].shape, device=args.device), torch.zeros(ll_neg_logits[0].shape, device=args.device)
            all_ll_pos_logits, all_ll_neg_logits = None, None
            all_ll_pos_labels, all_ll_neg_labels = None, None
            all_indices = None
            for i in range(len(datasets_information['domains'])):
                indices = np.where(pos_cd[:, i, :] != 0)  # [B,num_domain,L]
                if all_ll_pos_logits is None:
                    all_ll_pos_logits = ll_pos_logits[i]
                    all_ll_neg_logits = ll_neg_logits[i]
                    all_ll_pos_labels = ll_pos_labels
                    all_ll_neg_labels = ll_neg_labels
                    all_indices = indices
                else:
                    all_ll_pos_logits = torch.cat((all_ll_pos_logits, ll_pos_logits[i]), dim=1)
                    all_ll_neg_logits = torch.cat((all_ll_neg_logits, ll_neg_logits[i]), dim=1)
                    all_ll_pos_labels = torch.cat((all_ll_pos_labels, ll_pos_labels), dim=1)
                    all_ll_neg_labels = torch.cat((all_ll_neg_labels, ll_neg_labels), dim=1)
                    new_indices = (indices[0], indices[1] + i * pos_cd.shape[2])
                    all_indices = (np.concatenate((all_indices[0], new_indices[0]), axis=0),
                                   np.concatenate((all_indices[1], new_indices[1]), axis=0))
            ll_loss = bce_criterion(all_ll_pos_logits[all_indices], all_ll_pos_labels[all_indices])
            ll_loss += bce_criterion(all_ll_neg_logits[all_indices], all_ll_neg_labels[all_indices])

            # [B,num_domain,L]
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            lg_pos_labels, lg_neg_labels = torch.ones(lg_pos_logits[0].shape, device=args.device), torch.zeros(
                lg_neg_logits[0].shape, device=args.device)
            all_lg_pos_logits, all_lg_neg_logits = None, None
            all_lg_pos_labels, all_lg_neg_labels = None, None  # [B,L]
            all_indices = None
            for i in range(len(datasets_information['domains'])):
                indices = np.where(pos_oth_dom[:, i, :] != 0)
                if all_lg_pos_logits is None:
                    all_lg_pos_logits = lg_pos_logits[i]
                    all_lg_neg_logits = lg_neg_logits[i]
                    all_lg_pos_labels = lg_pos_labels
                    all_lg_neg_labels = lg_neg_labels
                    all_indices = indices
                else:
                    all_lg_pos_logits = torch.cat((all_lg_pos_logits, lg_pos_logits[i]), dim=1)  # [B,L]
                    all_lg_neg_logits = torch.cat((all_lg_neg_logits, lg_neg_logits[i]), dim=1)
                    all_lg_pos_labels = torch.cat((all_lg_pos_labels, lg_pos_labels), dim=1)
                    all_lg_neg_labels = torch.cat((all_lg_neg_labels, lg_neg_labels), dim=1)
                    new_indices = (indices[0], indices[1] + i * pos_oth_dom.shape[2])
                    all_indices = (np.concatenate((all_indices[0], new_indices[0]), axis=0),
                                   np.concatenate((all_indices[1], new_indices[1]), axis=0))
            lg_loss = bce_criterion(all_lg_pos_logits[all_indices], all_lg_pos_labels[all_indices])
            lg_loss += bce_criterion(all_lg_neg_logits[all_indices], all_lg_neg_labels[all_indices])

            global_indices = np.where(pos != 0)
            g_pos_labels, g_neg_labels = torch.ones(gg_pos_logits.shape, device=args.device), torch.zeros(gg_neg_logits.shape, device=args.device)
            gg_loss = bce_criterion(gg_pos_logits[global_indices], g_pos_labels[global_indices])
            gg_loss += bce_criterion(gg_neg_logits[global_indices], g_neg_labels[global_indices])


            behavior_regularizer_loss = model.behavior_regularizer(domain_switch_behavior, next_behavior)




            # print(args.ll_loss_weight * ll_loss)
            # add by weight
            loss = (args.ll_loss_weight * ll_loss + args.gg_loss_weight * gg_loss  +
                    args.behavior_regularizer_weight * behavior_regularizer_loss + args.lg_loss_weight * lg_loss)
            # print(loss)
            loss += cls_loss * args.cls_weight #add contrastive loss
            sum_loss += loss
            '''
            1
            torch.Size([392556, 5])
            2
            torch.Size([392556, 5])
            torch.Size([392556, 5])
            torch.Size([392556, 5])
            '''
            count1, count2 = 0,0
            for param in model.domain_invariant_item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.domain_specific_item_emb.parameters(): loss += args.l2_emb * torch.norm(param)



            loss.backward()
            optimizer.step()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        #train_loss.append(sum_loss.item())

        if epoch % args.interval == 0:
            args.training = False
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            #t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            flag = False  # whether the performance of the proposed model improves
            print(t_valid)
            sum_performance = 0
            for domain in datasets_information['domains']:
                sum_performance += (t_valid[domain][0] + t_valid[domain][1])
                ans_list[domain].append([t_valid[domain][0], t_valid[domain][1], t_valid[domain][2]])
                print(
                    'domain:%s, epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f, MRR@%d: %.4f)'
                    % (domain, epoch, T, args.top_n, t_valid[domain][0], args.top_n, t_valid[domain][1],
                       args.top_n, t_valid[domain][2]))

                if t_valid[domain][0] + t_valid[domain][1] > best_result[domain]:
                    flag = True
                    '''
                    # update results in all domain
                    for temp_domain in datasets_information['domains']:
                        best_result[temp_domain] = t_valid[temp_domain][0] + t_valid[temp_domain][1]
                        val_ndcg_10[temp_domain], val_hr_10[temp_domain], test_ndcg_10[temp_domain], test_hr_10[
                            temp_domain] \
                            = t_valid[temp_domain][0], t_valid[temp_domain][1], t_test[temp_domain][0], \
                        t_test[temp_domain][1]
                    '''


            if sum_best_performance <= sum_performance:
                sum_best_performance = sum_performance
            else:
                flag = 0 # model doesn't improve in all datasets

            if flag:
                # update results in all domain
                for temp_domain in datasets_information['domains']:
                    best_result[temp_domain] = t_valid[temp_domain][0] + t_valid[temp_domain][1]
                    val_ndcg_10[temp_domain], val_hr_10[temp_domain], val_mrr_10[temp_domain] = t_valid[temp_domain][0], t_valid[temp_domain][1], t_valid[temp_domain][2]
                count = 0
                folder = 'all'
                fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(information, args.num_epochs, args.lr,
                                     args.num_domain_shared_blocks + args.num_domain_specific_blocks, args.num_heads,
                                     args.hidden_units,
                                     args.maxlen)
                if True:
                    torch.save(model.state_dict(), os.path.join(notes, folder, fname))
            else:
                count += 1  # the performance of the proposed model doesn't improve in this evaluation

            t0 = time.time()
            model.train()
            print('loss = ', sum_loss)
            train_loss.append(sum_loss.item())
            args.training = True


        if count > args.early_stop:
            break

    model.load_state_dict(torch.load(os.path.join(notes, folder, fname), map_location=torch.device(args.device)))
    model.eval()
    t_test = evaluate(model, dataset, args)
    for temp_domain in datasets_information['domains']:
        test_ndcg_10[temp_domain], test_hr_10[temp_domain], test_mrr_10[temp_domain] = t_test[temp_domain][0], t_test[temp_domain][1], t_test[temp_domain][2]
    # save the best results in all domain
    best_result_domains = {}
    parameters_domains = {}


    for domain in datasets_information['domains']:

        df = pd.DataFrame(data=ans_list[domain],
                          columns=['val_NDCG@{0:}'.format(args.top_n), 'val_HR@{0:}'.format(args.top_n),
                                   'val_MRR@{0:}'.format(args.top_n)])
        df.to_csv(path_or_buf=os.path.join(notes, domain + '_' + args.train_dir, 'result{0:}.csv'.format(information)),
                  index=False)
        # deleting something results in errors
        # f.close()
        # different join
        with open(os.path.join(notes, domain + '_' + args.train_dir, 'args{0:}.txt'.format(information)), 'w') as f:
            f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
        f.close()
        sampler.close()


        parameters_in_this_domain = {
            'sequence_length_in_current_domain': len_domains[domain],
            'sequence_length_in_all_domains': len_domains['all']
        }
        parameters_domains[domain] = {
                    **hyper_parameters,**parameters_in_this_domain
        }
        with open(os.path.join(notes, domain + '_' + args.train_dir, '{0:}_parameters.json'.format(information)), 'w') as f:
            json.dump(
                parameters_domains[domain],f
            )

        with open(os.path.join(notes, domain + '_' + args.train_dir, '{0:}_results.json'.format(information)), 'w') as f:
            json.dump(
                {
                    'val_ndcg_10':val_ndcg_10[domain],
                    'val_hr_10':val_hr_10[domain],
                    'val_mrr_10':val_mrr_10[domain],
                    'test_ndcg_10':test_ndcg_10[domain],
                    'test_hr_10':test_hr_10[domain],
                    'test_mrr_10':test_mrr_10[domain]
                },f
            )
        best_result_domains[domain] = {}
        best_result_domains[domain]['val_ndcg_10'] = val_ndcg_10[domain]
        best_result_domains[domain]['val_hr_10'] = val_hr_10[domain]
        best_result_domains[domain]['val_mrr_10'] = val_mrr_10[domain]
        best_result_domains[domain]['test_ndcg_10'] = test_ndcg_10[domain]
        best_result_domains[domain]['test_hr_10'] = test_hr_10[domain]
        best_result_domains[domain]['test_mrr_10'] = test_mrr_10[domain]



    if not os.path.isdir(os.path.join(notes, 'all')):
        os.makedirs(os.path.join(notes, 'all'))


    with open(os.path.join(notes, 'all','{0:}_parameters.json'.format(information)),'w') as f:
        json.dump(parameters_domains,f)

    with open(os.path.join(notes, 'all','{0:}_results.json'.format(information)),'w') as f:
        json.dump(best_result_domains,f)

    table_result = pd.DataFrame(best_result_domains)
    table_result.to_csv(path_or_buf=os.path.join(notes, 'all' , 'table_result{0:}.csv'.format(information)),
                  index=False)

    df_train_loss = pd.DataFrame(train_loss)
    df_train_loss.to_csv(path_or_buf=os.path.join(notes, 'all' , 'train_loss{0:}.csv'.format(information)),
                  index=False)







    if not args.saved_the_model:
        #delate the model
        os.remove(os.path.join(notes, folder, fname))


    for domain in datasets_information['domains']:
    print('domain:%s, test (NDCG@%d: %.4f, HR@%d: %.4f, MRR@%d: %.4f)'
        % (domain, args.top_n, t_test[domain][0], args.top_n, t_test[domain][1],
           args.top_n, t_test[domain][2]))


    print("Done")


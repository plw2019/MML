import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class GRUEncoder(torch.nn.Module):
    def __init__(self, user_num, item_num, num_blocks, args):
        super(GRUEncoder, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        self.forward_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        # print('num_blocks:', num_blocks)
        # print('hidden_units:', args.hidden_units)
        self.gru = torch.nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units, num_layers=num_blocks)


    def forward(self, log_seqs, seqs):
        # the log_seqs denote the user interaction sequence
        # the seqs denote the embedding of the user interaction sequence
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        #[B,L,D]
        seqs = torch.transpose(seqs, 0, 1)
        #[L,B,D]
        seqs, hidden = self.gru(seqs)
        seqs = torch.transpose(seqs, 0, 1)

        seqs = self.forward_layernorms(seqs)
        seqs = self.forward_layers(seqs)

        return seqs



class MDR_local_global(torch.nn.Module):
    def __init__(self, user_num, item_num, datasets_information, args):
        super(MDR_local_global, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.domain_invariant_item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.domain_specific_item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0) # TO IMPROVE
        self.datasets_information = datasets_information
        self.dev = args.device
        self.mse = nn.MSELoss(reduction='sum')  # define mse for construction loss
        self.b_cos = True #catculate inner product after normalization
        self.temperature = args.temperature
        #seq [domain_id-1,seq] # every row indicates the index of the domain_id-1
        #data [B,domain_id-1,seq]
        self.mixed_seq_AE = GRUEncoder(self.user_num, self.item_num, args.num_blocks_mixed_seq, args)


        self.domain_specific_AE = torch.nn.ModuleList() # domain_id
        for _ in range(len(self.datasets_information['domains'])):
            self.domain_specific_AE.append(GRUEncoder(self.user_num, self.item_num, args.num_domain_specific_blocks, args))

        self.all_domain_maps = torch.nn.ModuleList()  # domain_id
        for _ in range(len(self.datasets_information['domains'])):
            self.all_domain_maps.append(torch.nn.Linear(args.hidden_units, args.hidden_units))

        self.map_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)


    def get_sequence_embedding(self, log_seqs, domain_mid = None):
        # get the embedding of the user interaction sequence
        # item_embedding + position_embedding
        # domain_mid equals domain - 1
        if domain_mid == 'domain_invariant':
            # domain_invariant
            seqs = self.domain_invariant_item_emb(torch.LongTensor(log_seqs).to(self.dev))
            seqs *= self.domain_invariant_item_emb.embedding_dim ** 0.5
        elif domain_mid == 'domain_specific':
            seqs = self.domain_specific_item_emb(torch.LongTensor(log_seqs).to(self.dev))
            seqs *= self.domain_specific_item_emb.embedding_dim ** 0.5


        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # # the position of the padding item is 0
        # positions[log_seqs == 0] = 0
        # print(positions)
        # seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        return seqs



    def forward(self, hybrid_seq, seq_cur_dom, pos_cur_dom, neg_cur_dom, hybrid_pos_seq, hybrid_neg_seq, arg_hybrid_seq, pos_oth_dom, neg_oth_dom, lg_dom):
        # calculate data
        # i denotes domain_id minuses one
        # [B,domain_num,L]
        log_feats = [None] * len(self.datasets_information['domains']) # the domain-specific preferences in each domain
        for i in range(len(self.datasets_information['domains'])):
            seqs = self.get_sequence_embedding(seq_cur_dom[:,i,:],'domain_specific')
            log_feats[i] = self.domain_specific_AE[i](seq_cur_dom[:,i,:], seqs) # [B,L,D]
            # calculate the output through the self-attention block

        # the domain determine the process of the calculation
        map_log_feats = [None] * len(self.datasets_information['domains'])
        for i in range(len(self.datasets_information['domains'])):
            for j in range(len(self.datasets_information['domains'])):
                domain_mask = torch.BoolTensor(lg_dom[:,i,:] == j + 1).to(self.dev)
                if j == 0:
                    map_log_feats[i] = self.all_domain_maps[j](log_feats[i]) * domain_mask.unsqueeze(-1)
                else:
                    map_log_feats[i] += self.all_domain_maps[j](log_feats[i]) * domain_mask.unsqueeze(-1)
            
            map_log_feats[i] = self.emb_dropout(map_log_feats[i])
            map_log_feats[i] = self.map_layernorm(map_log_feats[i])




        hybrid_seq_embs = self.get_sequence_embedding(hybrid_seq, 'domain_invariant')
        hybrid_feats = self.mixed_seq_AE(hybrid_seq, hybrid_seq_embs)

        #log_feats = log_feats + transferred_log_feats # the transferred information is leveraged
        # the domain determine the process of the calculation
        pos_cur_dom_embs, neg_cur_dom_embs = [None] * len(self.datasets_information['domains']), [None] * len(self.datasets_information['domains'])
        pos_oth_dom_embs, neg_oth_dom_embs = [None] * len(self.datasets_information['domains']), [None] * len(self.datasets_information['domains'])
        for i in range(len(self.datasets_information['domains'])):
            pos_cur_dom_embs[i] = self.domain_specific_item_emb(torch.LongTensor(pos_cur_dom[:,i,:]).to(self.dev))
            neg_cur_dom_embs[i] = self.domain_specific_item_emb(torch.LongTensor(neg_cur_dom[:,i,:]).to(self.dev))
            pos_oth_dom_embs[i] = self.domain_specific_item_emb(torch.LongTensor(pos_oth_dom[:,i,:]).to(self.dev))
            neg_oth_dom_embs[i] = self.domain_specific_item_emb(torch.LongTensor(neg_oth_dom[:,i,:]).to(self.dev))



        domain_invariant_hybrid_pos_embs = self.domain_invariant_item_emb(torch.LongTensor(hybrid_pos_seq).to(self.dev))
        domain_invariant_hybrid_neg_embs = self.domain_invariant_item_emb(torch.LongTensor(hybrid_neg_seq).to(self.dev))


        ll_pos_logits = [None] * len(self.datasets_information['domains'])
        ll_neg_logits = [None] * len(self.datasets_information['domains'])
        for i in range(len(self.datasets_information['domains'])):
            ll_pos_logits[i] = (log_feats[i] * pos_cur_dom_embs[i]).sum(dim=-1)
            ll_neg_logits[i] = (log_feats[i] * neg_cur_dom_embs[i]).sum(dim=-1)

        lg_pos_logits = [None] * len(self.datasets_information['domains'])
        lg_neg_logits = [None] * len(self.datasets_information['domains'])
        for i in range(len(self.datasets_information['domains'])):
            lg_pos_logits[i] = (map_log_feats[i] * pos_oth_dom_embs[i]).sum(dim=-1)
            lg_neg_logits[i] = (map_log_feats[i] * neg_oth_dom_embs[i]).sum(dim=-1)


        h_hybrid_pos_logits = (hybrid_feats * domain_invariant_hybrid_pos_embs).sum(dim=-1)  # [B,domain_num,L,D] -> [B,domain_num,L]
        h_hybrid_neg_logits = (hybrid_feats * domain_invariant_hybrid_neg_embs).sum(dim=-1)  # [B,domain_num,L,D] -> [B,domain_num,L]


        # predict items by considering domain-invariant preference and domain-specific preference
        # c denote concat, combine domain_invariant features and domain_specific features

        arg_hybrid_seq_embs = self.get_sequence_embedding(arg_hybrid_seq, 'domain_invariant')
        arg_hybrid_feats = self.mixed_seq_AE(arg_hybrid_seq, arg_hybrid_seq_embs)

        cls_loss = self.InfoNCE(hybrid_feats[:, -1, :], arg_hybrid_feats[:, -1, :])
        
        return ll_pos_logits, ll_neg_logits, h_hybrid_pos_logits, h_hybrid_neg_logits, lg_pos_logits, lg_neg_logits, cls_loss



    def predict(self, domain_id, log_seqs, hybrid_seqs, item_indices):

        # the domain determine the process of the calculation
        # seqs = self.get_sequence_embedding(log_seqs,'domain_specific')
        hybrid_seqs_embs = self.get_sequence_embedding(hybrid_seqs,'domain_invariant')
        # calculate the result the through the model
        # seqs= self.domain_specific_AE[domain_id-1](log_seqs, seqs)
        # log_feats = self.domain_shared_AE(log_seqs, seqs) #doesn't subscriptable
        hybrid_seqs_feats = self.mixed_seq_AE(hybrid_seqs, hybrid_seqs_embs)
        # print(log_seqs)
        # print(hybrid_seqs)
        # print(hybrid_seqs[hybrid_seqs_domains == domain_id])

        # print(local_preferrence_transferred_seqs)
        # print(hybrid_seqs)
        # this module is difficult, for it isn't aligned
        # the log_feats only contain the information in one domain
        # print(hybrid_seqs)
        # print(seq_single_domain)
        # the domain determine the process of the calculation
        seqs = self.get_sequence_embedding(log_seqs, 'domain_specific')
        # calculate the result the through the model
        local_log_feats = self.domain_specific_AE[domain_id - 1](log_seqs, seqs)
        # local_log_feats = self.domain_shared_AE(log_seqs, seqs)  # doesn't subscriptable
        # get item_embedding

        # print(transferred_log_feats)
        # print(last_id)
        # print(transferred_log_feats[:,last_id,:])

        # get item_embedding
        domain_invariant_item_embs = self.domain_invariant_item_emb(
            torch.LongTensor(item_indices).to(self.dev))  # (U, I, C) #[item_num,D]
        domain_specific_item_embs = self.domain_specific_item_emb(
            torch.LongTensor(item_indices).to(self.dev))


        #print(transferred_log_feats[:,last_id,:])

        final_item_embs = torch.cat((domain_invariant_item_embs,domain_specific_item_embs),dim=-1) #[N,d],[N,d] ->[N,2*d]
        final_feats = torch.cat( (hybrid_seqs_feats[:, -1, :], local_log_feats[:, -1, :]), dim=-1) #[1,d],[1,d] ->[1,2d]
        # get the preference for user about item
        # print(final_feat.shape)
        # print(item_embs.shape)
        logits = final_item_embs.matmul(final_feats.unsqueeze(-1)).squeeze(-1) #[1,item_num]
        # print(logits.shape)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        return logits


    def behavior_regularizer(self, domain_switch_behavior, next_behavior):
        domain_switch_behavior_embs = self.domain_invariant_item_emb(torch.LongTensor(domain_switch_behavior).to(self.dev)) # [B,num_dom, L] -> [B,num_dom,L,D]
        next_behavior_embs = self.domain_invariant_item_emb(torch.LongTensor(next_behavior).to(self.dev)) # [B,num_dom, L] -> [B,num_dom,L,D]
        regularizer_loss = self.mse(domain_switch_behavior_embs, next_behavior_embs)

        return regularizer_loss



    def InfoNCE(self, view1, view2):
        """
            Args:
                view1: (torch.Tensor - N x D)
                view2: (torch.Tensor - N x D)

            Return: Average InfoNCE Loss
            """
        if self.b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = torch.matmul(view1, view2.T)/ self.temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.sum()


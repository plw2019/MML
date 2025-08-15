import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from mamba_ssm import Mamba


class MambdaEncoder(torch.nn.Module):
    def __init__(self, user_num, item_num, num_layers, args):
        super(MambdaEncoder, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_size = args.hidden_units
        self.dropout_prob = args.dropout_rate
        self.num_layers = num_layers
        # Hyperparameters for Mamba block
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand

        self.batch_size = args.batch_size
        self.dim_size = args.hidden_units


        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])


    def forward(self, item_emb):
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = item_emb
        return seq_output





class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:  # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states








class MDR_local_global(torch.nn.Module):
    def __init__(self, user_num, item_num, datasets_information, args):
        super(MDR_local_global, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.domain_invariant_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_units, padding_idx=0)
        self.domain_specific_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0) # TO IMPROVE
        self.datasets_information = datasets_information
        self.dev = args.device
        self.mse = nn.MSELoss(reduction='sum')  # define mse for construction loss
        self.b_cos = True #catculate inner product after normalization
        self.temperature = args.temperature
        #seq [domain_id-1,seq] # every row indicates the index of the domain_id-1
        #data [B,domain_id-1,seq]
        self.mixed_seq_AE = MambdaEncoder(self.user_num, self.item_num, args.num_blocks_mixed_seq, args)

        self.domain_specific_AE = torch.nn.ModuleList() # domain_id
        for _ in range(len(self.datasets_information['domains'])):
            self.domain_specific_AE.append(MambdaEncoder(self.user_num, self.item_num, args.num_domain_specific_blocks, args))

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


        return seqs



    def forward(self, hybrid_seq, seq_cur_dom, pos_cur_dom, neg_cur_dom, hybrid_pos_seq, hybrid_neg_seq, arg_hybrid_seq, pos_oth_dom, neg_oth_dom, lg_dom):
        # calculate data
        # i denotes domain_id minuses one
        # [B,domain_num,L]
        log_feats = [None] * len(self.datasets_information['domains']) # the domain-specific preferences in each domain
        for i in range(len(self.datasets_information['domains'])):
            seqs = self.get_sequence_embedding(seq_cur_dom[:,i,:],'domain_specific')
            log_feats[i] = self.domain_specific_AE[i](seqs) # [B,L,D]
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
        hybrid_feats = self.mixed_seq_AE(hybrid_seq_embs)

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
        arg_hybrid_feats = self.mixed_seq_AE(arg_hybrid_seq_embs)

        cls_loss = self.InfoNCE(hybrid_feats[:, -1, :], arg_hybrid_feats[:, -1, :])
        
        return ll_pos_logits, ll_neg_logits, h_hybrid_pos_logits, h_hybrid_neg_logits, lg_pos_logits, lg_neg_logits, cls_loss



    def predict(self, domain_id, log_seqs, hybrid_seqs, item_indices):

        # the domain determine the process of the calculation
        # seqs = self.get_sequence_embedding(log_seqs,'domain_specific')
        hybrid_seqs_embs = self.get_sequence_embedding(hybrid_seqs,'domain_invariant')
        # calculate the result the through the model
        # seqs= self.domain_specific_AE[domain_id-1](log_seqs, seqs)
        # log_feats = self.domain_shared_AE(log_seqs, seqs) #doesn't subscriptable
        hybrid_seqs_feats = self.mixed_seq_AE(hybrid_seqs_embs)
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
        local_log_feats = self.domain_specific_AE[domain_id - 1](seqs)
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
        # The loss weight of behavior regularizer is affected by batch size, maxlen and datasets. If they are changed, the optimal weight might be changed.

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
        # The loss weight of contrastive loss is affected by batch size. If batch size is changed, the optimal weight might be changed.

        return -score.sum()


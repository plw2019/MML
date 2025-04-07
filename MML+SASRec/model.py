import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

#AttEncoder
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.scale = 1 / (self.d_k ** 0.5)

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # self.output_linear = nn.Linear(d_model, d_model)
        # no output linear in SASRec

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer=None, info=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if info is not None:
            info['input_seq' + str(layer)] = value
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        if info is not None:
            info['value_seq' + str(layer)] = value

        x, attn = self.attention(query, key, value, mask=mask, layer=layer, info=info)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        if info is not None:
            info['attn_seq' + str(layer)] = x

        # x =  self.output_linear(x)
        # no output linear in SASRec
        # if info is not None:
        #     info['output_seq' + str(layer)] = x
        return x

    def attention(self, query, key, value, mask=None, layer=None, info=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self.scale
        # print('================')
        # print(scores.shape)
        # print(mask.shape)
        if len(mask.shape) > 2:
            mask = mask.unsqueeze(1)
        # print(mask.shape)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # remove irrelevant information
        # print(mask)
        # print(p_attn)
        p_attn = p_attn * mask # remove irrelevant information
        # print(p_attn)
        #print(p_attn)
        if info is not None:
            info['attn_scores' + str(layer)] = p_attn

        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

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
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class CrossAttention(torch.nn.Module):
    def __init__(self, block_num, args):
        super(CrossAttention,self).__init__()

        self.dev = args.device
        self.block_num = block_num
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.cross_attention_layernorms = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(self.block_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiHeadedAttention(args.num_heads, args.hidden_units, args.dropout_rate)
            self.cross_attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs, seqs_embs, entire_log_seqs):
        '''

        :param log_seqs: the interacted sequence in the target domain
        :param seqs_embs: the interacted sequence embedding in all domains
        :param entire_log_seqs: the interacted sequence in all domains
        :return:
        '''
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # [B,L]
        all_timeline_mask = torch.BoolTensor(entire_log_seqs != 0).to(self.dev)  # [B,L] the value is true while the item isn't the padding item
        # if is zero, the answer is true
        # didn't pass the module
        # the interacted sequence contains the source domain and the target domain
        if self.block_num == 0:
            seqs_embs *= 0  # cover the answer
            #seqs_embs *= timeline_mask.unsqueeze(-1) # cover the answer
            return seqs_embs

        tl = seqs_embs.shape[1]  # time dim len for enforce causality
        attention_mask = torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # print(attention_mask)
        # print(log_seqs)
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        '''
        tensor([[ True, False, False, False, False, False, False, False, False, False],
        [ True,  True, False, False, False, False, False, False, False, False],
        [ True,  True,  True, False, False, False, False, False, False, False],
        [ True,  True,  True,  True, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]])
        '''

        # if is non-zero, the answer is false

        # torch.tile(last_log_feat.unsqueeze(0), dims=[user_embs.shape[0],1,1])
        # time_mask [B,1,L] -> [B,L,L]
        # # if has, zero
        target_mask = torch.tile(timeline_mask.unsqueeze(1), dims=[1,tl,1])
        target_mask2 = torch.tile(all_timeline_mask.unsqueeze(1), dims=[1,tl,1])
        # # print(target_mask)
        attention_mask = attention_mask * target_mask * target_mask2 # only true will be keep

        # print(log_seqs)
        # print(entire_log_seqs)
        # print(attention_mask)

        # print(attention_mask)

        # though the source seqs embs contain the padding embedding, the target seqs embs are useful
        target_seqs_embs = seqs_embs * ~timeline_mask.unsqueeze(-1)
        source_seqs_embs = seqs_embs * timeline_mask.unsqueeze(-1)

        # print(target_seqs_embs)
        # print(source_seqs_embs)
        # print(attention_mask.shape)
        # print(target_seqs_embs.shape)
        # print(source_seqs_embs.shape)

        for i in range(len(self.cross_attention_layers)):
            Q, K, V = target_seqs_embs, source_seqs_embs, source_seqs_embs
            Q = self.cross_attention_layernorms[i](Q)
            mha_outputs = self.cross_attention_layers[i](Q, K, V,
                                                   mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            if i == 0:
                target_seqs_embs = mha_outputs # doesn't contain the information in target domain
            elif i > 0:
                target_seqs_embs = target_seqs_embs + mha_outputs # cross-attention has many blocks
            # seqs = torch.transpose(seqs, 0, 1)

            # print(target_seqs_embs)
            target_seqs_embs = self.forward_layernorms[i](target_seqs_embs)
            target_seqs_embs = self.forward_layers[i](target_seqs_embs)
            # print(target_seqs_embs.shape)
            # print((~timeline_mask).shape)
            target_seqs_embs *= ~timeline_mask.unsqueeze(-1)


        transfered_target_seqs_embs = self.last_layernorm(target_seqs_embs)  # (U, T, C) -> (U, -1, C)
        transfered_target_seqs_embs *= ~timeline_mask.unsqueeze(-1)  # if this is padding item, the log_feat is zero embedding
        # print(transfered_target_seqs_embs)

        return transfered_target_seqs_embs



class AttEncoder(torch.nn.Module):
    def __init__(self, user_num, item_num, block_num, args):
        super(AttEncoder, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.block_num = block_num # the number of the self-attention layer

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)


        for _ in range(self.block_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  MultiHeadedAttention(args.num_heads,args.hidden_units,args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)




    def forward(self, log_seqs, seqs):
        # didn't pass the module
        if self.block_num == 0:
            return seqs
        # the log_seqs denote the user interaction sequence
        # the seqs denote the embedding of the user interaction sequence
        
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # print(attention_mask.shape)
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        '''
        tensor([[ True, False, False, False, False, False, False, False, False, False],
        [ True,  True, False, False, False, False, False, False, False, False],
        [ True,  True,  True, False, False, False, False, False, False, False],
        [ True,  True,  True,  True, False, False, False, False, False, False],
        [ True,  True,  True,  True,  True,  True, False, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]])
        '''


        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            Q, K, V = seqs, seqs, seqs
            Q = self.attention_layernorms[i](Q)
            mha_outputs = self.attention_layers[i](Q, K, V,
                                            mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            # print('==============')
            # print(mha_outputs.shape)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        log_feats *=  ~timeline_mask.unsqueeze(-1) # if this is padding item, the log_feat is zero embedding

        return log_feats

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
        self.mixed_seq_AE = AttEncoder(self.user_num, self.item_num, args.num_blocks_mixed_seq, args)
        self.domain_shared_AE = AttEncoder(self.user_num, self.item_num, args.num_domain_shared_blocks, args)

        self.domain_specific_AE = torch.nn.ModuleList() # domain_id
        for _ in range(len(self.datasets_information['domains'])):
            self.domain_specific_AE.append(AttEncoder(self.user_num, self.item_num, args.num_domain_specific_blocks, args))

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


        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # the position of the padding item is 0
        positions[log_seqs == 0] = 0
        # print(positions)
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
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


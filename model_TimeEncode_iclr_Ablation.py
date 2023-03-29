import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    #### future_mask is a np object. Convert it to tensor
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def attention(query, key, value, rel, l1, l2, timestamp, ablation_Dict, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    ## comment out rel
    scores_temp = torch.matmul(query, key.transpose(-2, -1))
     ### Simply type-cast your boolean mask to an integer mask, followed by float to bring the mask to the same type as img. Perform element-wise multiplication afterwards.
    ## https://stackoverflow.com/questions/58521595/masking-tensor-of-same-shape-in-pytorch
    rel = rel * mask.to(torch.float) # future masking of correlation matrix.
    rel_attn = rel.masked_fill(rel == 0, -10000)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    
    #rel_attn = 0


    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

        #### neg exponential of timedelta
        time_stamp= torch.exp(-torch.abs(timestamp.float()))
        #
        time_stamp=time_stamp.masked_fill(mask,-np.inf)


    prob_attn = F.softmax(scores, dim=-1)
    time_attn = F.softmax(time_stamp,dim=-1)

    ## CHECK ABLATION_DICT
    if ablation_Dict["time_include"] == True:

        prob_attn = (1-l2)*prob_attn + l2*time_attn
    

 
    if ablation_Dict["qrelation_include"] == True:
        prob_attn = (1-l1)*prob_attn + (l1)*rel_attn

    
    if ablation_Dict["time_include"] == False:
        prob_attn = l2*prob_attn + 0

    
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    
    return torch.matmul(prob_attn, value), prob_attn



class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, rel, l1, l2, timestamp,  pos_key_embeds, pos_value_embeds, ablation_Dict, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        ## COMMENT out rel
        #print("in MultiHeadedAttention Forward rel type ", type(rel), " shape ", rel.shape)
        ## results in dimensions = 4: batch_size (1st param of repeat) x num_heads (2nd param repeat == num repeat times) x org_dim(= 2) x org_dim(=3). repeat(1, self.num_heads, 1,1 )
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)

        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
         
        out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, ablation_Dict, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn

class PositionalEmbedding(nn.Module):
    # formula https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model

    def __init__(self, embedding_size, max_seq_len = 80):
      ###  max_seq_len is the maximum word in  a sentence. the default value is 80 but in this code, we pass a value of 20

        super().__init__()
        self.d_model = embedding_size
        self.d_model_inter = 4*embedding_size#3*embedding_size

        pe = torch.zeros(max_seq_len, embedding_size)

        for pos in range(max_seq_len):
            for i in range(0, embedding_size, 2):

                div_term = math.exp(i * -math.log(10000.0) / self.d_model)

                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))#math.sin(pos * div_term )
                pe[pos, i + 1] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))#math.cos(pos * div_term)

        pe = pe.unsqueeze(0)
        #self.register_buffer('pe', self.pe)
        self.weight = nn.Parameter(pe, requires_grad=False)


        ## pe inter

        pe_inter = torch.zeros(max_seq_len, self.d_model_inter)

        for pos in range(max_seq_len):
            for i in range(0, self.d_model_inter, 2):

                div_term = math.exp(i * -math.log(10000.0) / self.d_model_inter)

                pe_inter[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model_inter)))#math.sin(pos * div_term )
                pe_inter[pos, i + 1] = math.sin(pos / (10000 ** ((2 * i)/self.d_model_inter)))#math.cos(pos * div_term)

        pe_inter = pe_inter.unsqueeze(0)
        #self.register_buffer('pe', self.pe)
        self.weight_inter = nn.Parameter(pe_inter, requires_grad=False)
    
    def forward(self, x, is_interaction):
      
        ### if is_interaction == true, then shift pe 1 ri8 position
        if is_interaction == 1:

            ## right shifted positions. So, for max_pos 15, temp_shifted_pe will be 14
            temp_shifted_pe = self.weight_inter[:,:(x.size(1)-1)]
            ## init [1 x embed_size] matrix filled with zero
            shift_val =  torch.zeros(1, self.d_model_inter)
            ## extend to 3 dimension. shift_val.size() = [1 x 1 x embed_size]
            shift_val = shift_val.unsqueeze(0)
            #print("shifted ", temp_shifted_pe)
            #print("shape shifted pe ", temp_shifted_pe.size(), "shify_ val", shift_val.size())
            cat_test = torch.cat([shift_val, temp_shifted_pe], dim = 1)
            #print("cat ", cat_test, " ", cat_test.size())
            temp_pe = cat_test

         
        else:

            temp_pe =  self.weight[:,:x.size(1)]

        ## repeat to match batch dimension of query
        repeated_pe = temp_pe.repeat(x.size(0), 1, 1)

        return repeated_pe



class SRL_KT(nn.Module):
    def __init__(self, num_items,  embed_size, num_attn_layers, num_heads,
                  max_pos, drop_prob, id_to_orgDict, USE_dict, ablation_Dict):
        """Self-attentive knowledge tracing.
        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(SRL_KT, self).__init__()
        self.embed_size = embed_size
        
        self.num_classes = 2

        ## add dict
        self.id_to_orgDict = id_to_orgDict
        self.USE_dict = USE_dict
        
        
        
        print("init SRL KT")
        self.item_embeds = nn.Embedding(num_items + 1, embed_size, padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        ### item_type == 0 is padding index
        self.item_type_embeds = nn.Embedding(4, embed_size , padding_idx=0)

        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        ### num_classes
        #self.lin_out = nn.Linear(embed_size, self.num_classes)
        self.lin_out = nn.Linear(embed_size, 1)
        ### l1 and l2 random arrays
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))

        self.ablation_dict = ablation_Dict

        #self.lin_in_interaction = nn.Linear(3*embed_size, embed_size)
        ## pe dim 4
        self.lin_in_interaction = nn.Linear(4*embed_size, embed_size)
        self.lin_in_interaction_ablation = nn.Linear(3*embed_size, embed_size)
        ## init pe
        self.pe = PositionalEmbedding(embed_size, max_pos)

        self.final_out = nn.Sequential(
            nn.Linear(2*embed_size, embed_size), nn.ReLU(), nn.Dropout(p=drop_prob),
            #nn.Linear(embed_size, 256), nn.ReLU(), nn.Dropout(p=drop_prob),
            nn.Linear(embed_size, 1)
        )

        self.final_respout = nn.Sequential(
            nn.Linear(3*embed_size, embed_size), nn.ReLU(), nn.Dropout(p=drop_prob),
            nn.Linear(embed_size, 1) )

   
    
    def get_inputs(self, item_inputs, label_inputs, types_, qresp_inputs, ablation_Dict):
        #temp = nn.Embedding(36170, 200, padding_idx=0)
        #print("before embedding ",item_inputs.size(), " item_inputs ", item_inputs)

        all_inp_embedding = []
        
        for i in range(len(item_inputs)):
            ### process per row
            curr_row = item_inputs[i]
            #print("curr row ", curr_row)

            row_embedding = []

            for j in range(len(curr_row)):

                curr_ij_elem = curr_row[j]
                ### convert to int
                curr_ij_elem = curr_ij_elem.item()
                #print("row ", i, "col j", j, "row ij ", curr_ij_elem)

                ## 1st element is dummy 0 input
                
                curr_item_embedding = np.zeros(self.embed_size)

                if j > 0:
                    item_org_index = self.id_to_orgDict[curr_ij_elem]
                    curr_item_embedding = np.array(self.USE_dict[item_org_index])
                    ### convert tf.tensor to array
                    
                #curr_item_embedding = torch.from_numpy(curr_item_embedding)
                row_embedding.append(curr_item_embedding)

            ### done with all timestamps in current row. append to all_inp_embedding
            all_inp_embedding.append(row_embedding)

        ### convert to tensor
        all_inp_embedding = np.array(all_inp_embedding)
        all_inp_embedding = torch.from_numpy(all_inp_embedding).float()
        
        item_inputs = all_inp_embedding

        
    
        label_inputs_1 = label_inputs.unsqueeze(-1).repeat(1, 1, self.embed_size)
        qresp_inputs_1 = qresp_inputs.unsqueeze(-1).repeat(1, 1, self.embed_size)
        ## added from model_srl
        type_inputs_1 = self.item_type_embeds(types_)
        


        
        inputs = torch.cat([item_inputs, item_inputs, item_inputs, item_inputs], dim=-1)  

        ## CHECK ablation
        if ablation_Dict["qrespCos_include"] == True:

            inputs[..., self.embed_size:2*self.embed_size] = type_inputs_1#types_
            inputs[..., 2*self.embed_size:3*self.embed_size] = qresp_inputs_1
            inputs[..., 3*self.embed_size:] = label_inputs_1

        else:
            ### modify input vector
            inputs = torch.cat([item_inputs, item_inputs, item_inputs], dim=-1)  
            inputs[..., self.embed_size:2*self.embed_size] = type_inputs_1#types_
            inputs[..., 2*self.embed_size:3*self.embed_size] = label_inputs_1
            #inputs[..., 3*self.embed_size:] = label_inputs_1



        if ablation_Dict["time_include"] == False:
            ## include pe
            pe_inputs = self.pe(inputs, is_interaction = 1)
            inputs = inputs + pe_inputs

        
        return inputs

 
    def get_query(self, item_ids):
    
        all_inp_embedding = []
        
        for i in range(len(item_ids)):
            ### process per row
            curr_row = item_ids[i]
            #print("curr row ", curr_row)

            row_embedding = []

            for j in range(len(curr_row)):

                curr_ij_elem = curr_row[j]
                ### convert to int
                curr_ij_elem = curr_ij_elem.item()
                
                ## NOT 1st element is dummy 0 input
                item_org_index = self.id_to_orgDict[curr_ij_elem]
                curr_item_embedding = np.array(self.USE_dict[item_org_index])
                ### convert tf.tensor to array
                    
                row_embedding.append(curr_item_embedding)

            ### done with all timestamps in current row. append to all_inp_embedding
            all_inp_embedding.append(row_embedding)

        ### convert to tensor
        all_inp_embedding = np.array(all_inp_embedding)
        all_inp_embedding = torch.from_numpy(all_inp_embedding).float()

        query = all_inp_embedding

        #print("query inputs", len(query), "shape ", query.size())

        return query

    def forward(self, item_inputs, label_inputs, type_inputs, item_ids, rel, timestamp, item_types, qresp_inputs):
        types_ = self.item_type_embeds(item_types)
        ### add 2 embeddings
        ##test_types = query + types_
        
        ## bcz all inputs are ri8 shifted one cell, so type_inputs will be correct argument instead of types_
        inputs = self.get_inputs(item_inputs, label_inputs, type_inputs, qresp_inputs, self.ablation_dict)

        ### lin_in has input size 2*embed_size == bcz ques+ score are concatenated to produce 2*embed_size vector. The output is embed size dimension
        ## to match with dimension of key and value. RKT uses interaction for both key and value
        #inputs = F.relu(self.lin_in(inputs))
        if self.ablation_dict["qrespCos_include"] == True:
            inputs = F.relu(self.lin_in_interaction(inputs))

        if self.ablation_dict["qrespCos_include"] == False:
            inputs = F.relu(self.lin_in_interaction_ablation(inputs))
        
       

        query_embed = self.get_query(item_ids)

        ## to concatenate later
        query = query_embed 

       
        mask = future_mask(inputs.size(-2))

        if inputs.is_cuda:
            mask = mask.cuda()

        outputs, attn  = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, timestamp, 
                                                   self.pos_key_embeds, self.pos_value_embeds, self.ablation_dict, mask)
        outputs = self.dropout(outputs)
        
        ## iterate through multi-head attention from INDEX 1 to |heads|? INDEX 0 is processed above ^
        for l in self.attn_layers[1:]:
            ### CHECK param ordering : timestamp
            #residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, self.encode_pos, timestamp, self.pos_key_embeds,
            #             self.pos_value_embeds, mask)
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2,  timestamp, self.pos_key_embeds,
                         self.pos_value_embeds, self.ablation_dict, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        ### lin_out is a linear layer: takes input of embed_size (== size of query/key/value) and outputs 1 (prediction label)
        ## For AL, the output will be num_class (prediction label) with softmax
        #return self.lin_out(outputs), attn

        ### final_out: takes input of 2*embed_size (== size of query/key/value) and outputs 1 (prediction label)
        concat_q = torch.cat([outputs, query_embed], dim=-1)
        output_ = self.final_out(concat_q)

        return output_, attn
        

      
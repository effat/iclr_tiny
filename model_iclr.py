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
    
    
     
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

        #### neg exponential of timedelta
        time_stamp= torch.exp(-torch.abs(timestamp.float()))
        time_stamp=time_stamp.masked_fill(mask,-np.inf)


    prob_attn = F.softmax(scores, dim=-1)
    time_attn = F.softmax(time_stamp,dim=-1)
    prob_attn = (1-l2)*prob_attn+l2*time_attn
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)

    if ablation_Dict["qrelation_include"] == True:
        rel = rel * mask.to(torch.float) # future masking of correlation matrix.
        rel_attn = rel.masked_fill(rel == 0, -10000)
        rel_attn = nn.Softmax(dim=-1)(rel_attn)

        prob_attn = (1-l1)*prob_attn + (l1)*rel_attn

    
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

    def forward(self, query, key, value, rel, l1, l2, timestamp, ablation_Dict, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

       
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)

        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)


        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
         
        out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, ablation_Dict, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn





class SRL_KT(nn.Module):
    def __init__(self, embed_size, num_attn_layers, num_heads,
                  max_pos, drop_prob, id_to_orgDict, USE_dict, ablation_Dict):
        """ Reading and Learning Activities knowledge tracing.
        Arguments:
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
            id_to_orgDict: Re-indexed ID to original question ID in the dataset mapping. Original question ID in the dataset is re-indexed to ), 1, 2, ...
            USE_dict: ID to universal sentence encoder mapping
            ablation_Dict: Ablation component
        """
        super(SRL_KT, self).__init__()
        self.embed_size = embed_size
        
        self.num_classes = 2

        ## add dict
        self.id_to_orgDict = id_to_orgDict
        self.USE_dict = USE_dict
        
        
        
        print("init SRL KT")
        
    
        self.item_type_embeds = nn.Embedding(4, embed_size , padding_idx=0)

        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        
        
        self.lin_out = nn.Linear(embed_size, 1)
        ### l1 and l2 random arrays
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))

        self.ablation_dict = ablation_Dict
        ### interaction vector size
        self.interaction_vector_size = 4

        
        self.lin_in_interaction = nn.Linear(self.interaction_vector_size*embed_size, embed_size)
        
       
        self.final_out = nn.Sequential(
            nn.Linear(2*embed_size, embed_size), nn.ReLU(), nn.Dropout(p=drop_prob),
            #nn.Linear(embed_size, 256), nn.ReLU(), nn.Dropout(p=drop_prob),
            nn.Linear(embed_size, 1)
        )

        '''
        COMMENTED OUT 

        self.final_respout = nn.Sequential(
            nn.Linear(3*embed_size, embed_size), nn.ReLU(), nn.Dropout(p=drop_prob),
            nn.Linear(embed_size, 1) )

        

        '''

    

   
    
    def get_inputs(self, item_inputs, label_inputs, types_, qresp_inputs, ablation_Dict):
        
        all_inp_embedding = []
        
        for i in range(len(item_inputs)):
            ### process per row
            curr_row = item_inputs[i]

            row_embedding = []

            for j in range(len(curr_row)):

                curr_ij_elem = curr_row[j]
                ### convert to int
                curr_ij_elem = curr_ij_elem.item()
                
                
                curr_item_embedding = np.zeros(self.embed_size)

                if j > 0:
                    item_org_index = self.id_to_orgDict[curr_ij_elem]
                    curr_item_embedding = np.array(self.USE_dict[item_org_index])
                    ### convert tf.tensor to array
                    
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

       

        inputs[..., self.embed_size:2*self.embed_size] = type_inputs_1#types_
        inputs[..., 2*self.embed_size:3*self.embed_size] = qresp_inputs_1
        inputs[..., 3*self.embed_size:] = label_inputs_1

        
        
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
       
        inputs = self.get_inputs(item_inputs, label_inputs, type_inputs, qresp_inputs, self.ablation_dict)

    
        #if self.ablation_dict["qrespCos_include"] == True:
        # lin_in_interaction vector size adjusted based on self.ablation_dict["qrespCos_include"] == True
        inputs = F.relu(self.lin_in_interaction(inputs))

        
        
       

        query_embed = self.get_query(item_ids)

        ## to concatenate later
        query = query_embed 

       
        mask = future_mask(inputs.size(-2))

        if inputs.is_cuda:
            mask = mask.cuda()

        outputs, attn  = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, timestamp, 
                                                    self.ablation_dict, mask)
        outputs = self.dropout(outputs)
        
        ## iterate through multi-head attention from INDEX 1 to |heads|? INDEX 0 is processed above ^
        for l in self.attn_layers[1:]:
            ### CHECK param ordering : timestamp
           
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, timestamp, self.ablation_dict, mask)
            outputs = self.dropout(outputs + F.relu(residual))

     
        ### final_out: takes input of 2*embed_size (== size of query/key/value) and outputs 1 (prediction label)
        concat_q = torch.cat([outputs, query_embed], dim=-1)
        output_ = self.final_out(concat_q)

        return output_, attn
        

      
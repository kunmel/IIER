import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from transformers import RobertaPreTrainedModel,RobertaModel

def init_weights(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.Dropout):
        pass 

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, bias, dropout_rate=0.):
        super(Linear, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias, dropout_rate=0.):
        super(MLP, self).__init__()
        self.linear1 = Linear(input_dim, hidden_dim, bias, dropout_rate)
        self.linear2 = Linear(hidden_dim, output_dim, bias, dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x
    
class Neighbour_scorer(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)
        self.args = args
        self.roberta = RobertaModel(config=config)

        self.MLP = MLP(input_dim=4*args.D, hidden_dim=args.D, output_dim=1, bias=True, dropout_rate=args.mlp_dropout_rate)
        self.f_edges = MLP(input_dim=3, hidden_dim=args.D, output_dim=args.D_2, bias=True, dropout_rate=args.mlp_dropout_rate)
        self.MLP.apply(init_weights)
        self.f_edges.apply(init_weights)
        
    def forward(self, q_input_ids, q_attention_mask, n_input_ids, n_attention_mask, s_input_ids, s_attention_mask,keywords_input_ids, keywords_attention_mask, path_input_ids, path_attention_mask, valid_edges, sim_scores, keywords_mask):
        
        hq = self.roberta(q_input_ids, attention_mask=q_attention_mask)[0][:, 0, :]
        hn = self.roberta(n_input_ids, attention_mask=n_attention_mask)[0][:, 0, :] 

        h_path = self.roberta(path_input_ids, attention_mask=path_attention_mask)[0][:, 0, :] 
        h_edges = self.f_edges(valid_edges)
        scores = self.MLP(torch.cat([hq, h_path, hn, h_edges], dim=1))
        return scores
        
    def compute_loss(self, scores, labels):
        return F.binary_cross_entropy_with_logits(scores, labels.float())

    
    
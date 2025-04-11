import logging
import utils
import torch
from data_reader import transfer_edge_type

logger = logging.getLogger(__name__)
utils.init_logger()

def predict(model, args, example):
    hqs, hns, hss, valid_edges, sim_scores, h_keywords, keywords_mask = [], [], [], [], [], [], []
    for node_pair_info in example:
        hqs.append(node_pair_info["h_query"])
        hns.append(node_pair_info["h_neighbour"])
        hss.append(node_pair_info["h_seed"])
        valid_edges.append(transfer_edge_type(node_pair_info["edges"]))
        if "sim_scores" in node_pair_info:
            sim_scores.append(node_pair_info["sim_scores"])
        else:
            sim_scores.append(float(0.0))
        if "h_keywords" in node_pair_info:
            keywords_mask.append([1 for _ in node_pair_info["h_keywords"]] + [0] * (5 - len(node_pair_info["h_keywords"]))) 
            h_keywords.append(node_pair_info["h_keywords"] + [args.mock_embedding] * (5 - len(node_pair_info["h_keywords"])))
        else:
            keywords_mask.append([0] * 5) 
            h_keywords.append([args.mock_embedding] * 5)
            hqs = torch.tensor(hqs, dtype=torch.long)
            hns = torch.tensor(hns, dtype=torch.long)
            hss = torch.tensor(hss, dtype=torch.long)
            valid_edges = torch.tensor(valid_edges, dtype=torch.long)
            sim_scores = torch.tensor(sim_scores, dtype=torch.float32).unsqueeze(1)
            h_keywords = torch.tensor(h_keywords, dtype=torch.long)
            keywords_mask = torch.tensor(keywords_mask, dtype=torch.long)
    with torch.no_grad():
        scores = model(hqs, hns, hss, valid_edges, sim_scores, h_keywords, keywords_mask)
        scores.detach().cpu().numpy()
        return scores.index(max(scores))
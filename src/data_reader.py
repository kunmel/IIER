import os
import json
import logging
import torch
import config
import random
from torch.utils.data import TensorDataset,  random_split
from transformers import RobertaTokenizer
import utils

logger = logging.getLogger(__name__)
utils.init_logger()

class DataProcessor(object):
    def __init__(self, args):
        self.args = args
        self.input_file = args.train_data_file

    def create_example(self, args):
        with open(self.input_file, "r") as f:
            examples = f.readlines()

        logger.info("***** Running data reader *****")
        if (os.path.exists(os.path.join(self.args.cache_data_path, "data"))):
            logger.info("***** Loading cached data *****")
            dataset = torch.load(os.path.join(self.args.cache_data_path, "data"))

            train_dataset, val_dataset, test_dataset = random_split(dataset, self.args.data_split)
            logger.info("***** Cached data loaded*****")
            return train_dataset, val_dataset, test_dataset
        logger.info("***** No cached data *****")
        os.makedirs(self.args.cache_data_path, exist_ok=True)
        valid_edges, labels= [], []
        q_input_ids, q_attention_mask, n_input_ids, n_attention_mask, path_input_ids, path_attention_mask = [], [], [], [], [], []

        tokenizer = RobertaTokenizer.from_pretrained(self.args.model_name_or_path)
        for idx, example in enumerate(examples):
            if idx < self.args.train_data_start_id:
                continue
            if idx >= self.args.train_data_end_id:
                break
            example = json.loads(example)
            current_labels = []
            valid_examples = make_valid_example(example)
            for inner_idx, node_pair_info in enumerate(valid_examples):
                if node_pair_info in example["positive_data"]:
                    current_labels.append(1)
                else:
                    current_labels.append(0)
                tokenized_q = tokenizer(node_pair_info["query_text"], truncation=True, padding='max_length', max_length=self.args.text_max_len, add_special_tokens=True, return_tensors='pt')
                q_input_ids.append(tokenized_q["input_ids"])
                q_attention_mask.append(tokenized_q["attention_mask"])

                tokenized_n = tokenizer(node_pair_info["neighbour_text"], truncation=True, padding='max_length', max_length=self.args.text_max_len, add_special_tokens=True, return_tensors='pt')
                n_input_ids.append(tokenized_n["input_ids"])
                n_attention_mask.append(tokenized_n["attention_mask"])

                tokenized_path = tokenizer(node_pair_info["path_text"], truncation=True, padding='max_length', max_length=self.args.text_max_len, add_special_tokens=True, return_tensors='pt')
                path_input_ids.append(tokenized_path["input_ids"])
                path_attention_mask.append(tokenized_path["attention_mask"])
                
            
                valid_edges.append(transfer_edge_type(node_pair_info))
                
            labels += current_labels
        q_input_ids = torch.cat(q_input_ids, dim=0)
        q_attention_mask = torch.cat(q_attention_mask, dim=0)

        n_input_ids = torch.cat(n_input_ids, dim=0)
        n_attention_mask = torch.cat(n_attention_mask, dim=0)

        path_input_ids = torch.cat(path_input_ids, dim=0)
        path_attention_mask = torch.cat(path_attention_mask, dim=0)
        
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        valid_edges = torch.tensor(valid_edges, dtype=torch.float32)

        show_size(q_input_ids, q_attention_mask, 
            n_input_ids, n_attention_mask, 
            path_input_ids, path_attention_mask, 
            valid_edges, labels)
        
        dataset = TensorDataset(
            q_input_ids, q_attention_mask, 
            n_input_ids, n_attention_mask, 
            path_input_ids, path_attention_mask, 
            valid_edges, labels)

        train_dataset, val_dataset, test_dataset = random_split(dataset, self.args.data_split)

        torch.save(dataset, os.path.join(self.args.cache_data_path, "data"))

        logger.info("***** Data reader done*****")
        return train_dataset, val_dataset, test_dataset
        
def make_valid_example(example):
    data_dict = {}
    data_list = []
    for positive_example in example["positive_data"]:
        data_list.append(positive_example)
        data_dict[(positive_example["node_id"], positive_example["neighbor_id"])] = 0
    random.shuffle(example["negative_data"])
    for negative_example in example["negative_data"]:
        if (negative_example["node_id"], negative_example["neighbor_id"]) in data_dict and data_dict[(negative_example["node_id"], negative_example["neighbor_id"])] < config.negative_ratio:
            data_list.append(negative_example)
            data_dict[(negative_example["node_id"], negative_example["neighbor_id"])] += 1
    return data_list
    

def transfer_edge_type(edge_info):
    r = [0, 0, 0]
    if "structure_adjacent" in edge_info["edges"]:
        r[0] = 1
    if "semantic_similar_1" in edge_info["edges"]:
        r[1] = edge_info["sim_score"]
    if "same_keyword" in edge_info["edges"]:
        r[2] = min(len(edge_info["text_keywords"]), 5)
        
    return r
 
def show_size(q_input_ids, q_attention_mask, 
            n_input_ids, n_attention_mask, 
            path_input_ids, path_attention_mask, 
            valid_edges, labels):
    logger.info(f"q_input_ids: {q_input_ids.size()}")
    logger.info(f"q_attention_mask: {q_attention_mask.size()}")
    logger.info(f"n_input_ids: {n_input_ids.size()}")
    logger.info(f"n_attention_mask: {n_attention_mask.size()}")
    logger.info(f"path_input_ids: {path_input_ids.size()}")
    logger.info(f"path_attention_mask: {path_attention_mask.size()}")
    logger.info(f"valid_edges: {valid_edges.size()}")
    logger.info(f"labels: {labels.size()}")
    
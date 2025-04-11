import config
import json
import utils
import pickle
import math
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import RobertaTokenizer

model_kwargs = {'device': 'cuda:7'}
embeddings = HuggingFaceEmbeddings(model_name="path2model", model_kwargs=model_kwargs)

def cosine_similarity(vec1, vec2):
    dot_product = sum(p*q for p, q in zip(vec1, vec2))
    magnitude_vec1 = math.sqrt(sum([x**2 for x in vec1]))
    magnitude_vec2 = math.sqrt(sum([x**2 for x in vec2]))
    if magnitude_vec1 * magnitude_vec2 == 0:
        return 0
    return dot_product / (magnitude_vec1 * magnitude_vec2)

    
def find_optimal_elements(data, target_strings, max_score):
    if len(data) == 0:
        print("*************len == 0 in find_optimal_elements")
    target_set = set(target_strings)
    best_cover = set()
    best_score = max_score
    best_keywords = set()

    for element, element_data in data.items():
        element_keywords = set(element_data['keywords'])
        element_score = element_data['score']

        if len(element_keywords & target_set) > len(best_keywords & target_set) or \
           (len(element_keywords & target_set) == len(best_keywords & target_set) and element_score < best_score):
            best_cover = {element}
            best_score = element_score
            best_keywords = element_keywords

    remaining_targets = target_set - set(data[list(best_cover)[0]]['keywords'])
    while remaining_targets:
        best_element = None
        best_new_cover = set()
        best_new_score = max_score
        best_new_keywords = set()

        for element, element_data in data.items():
            if element in best_cover:
                continue

            element_keywords = set(element_data['keywords'])
            new_cover = best_cover | {element}
            new_score = element_data['score']
            
            if len(element_keywords & remaining_targets) == 0:
                continue

            if len(element_keywords & remaining_targets) > len(best_new_keywords & remaining_targets) or \
               (len(element_keywords & remaining_targets) == len(best_new_keywords & remaining_targets) and new_score < best_new_score):
                best_element = element
                best_new_cover = new_cover
                best_new_score = new_score
                best_new_keywords = element_keywords

        if best_element is None:
            break

        best_cover = best_new_cover
        remaining_targets = remaining_targets - best_new_keywords

    return list(best_cover)



def select_neighbour(args, model, G, query, node_id, path, seen_nodes):
    neighbour_ids = list(G.neighbors(node_id))
    scores = []
    path_text = " ".join(path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    tokenized_q = tokenizer(query, truncation=True, padding='max_length', max_length=args.text_max_len, add_special_tokens=True, return_tensors='pt')
    tokenized_path = tokenizer(path_text, truncation=True, padding='max_length', max_length=args.text_max_len, add_special_tokens=True, return_tensors='pt')
    q_input_ids = tokenized_q["input_ids"]
    q_attention_mask = tokenized_q["attention_mask"]
    path_input_ids = tokenized_path["input_ids"]
    path_attention_mask = tokenized_path["attention_mask"]
    unseen_neighbour_ids = []
    q_input_ids_list, q_attention_mask_list, n_input_ids_list, n_attention_mask_list, path_input_ids_list, path_attention_mask_list, valid_edges_list = [], [], [], [], [], [], []
    for neighbour_id in neighbour_ids:
        if neighbour_id in seen_nodes:
            continue
        unseen_neighbour_ids.append(neighbour_id)
        neighbour_info = G.nodes[neighbour_id]
        edges = dict(G.get_edge_data(node_id, neighbour_id))
        r = [0, 0, 0]
        for edge_info in edges.values():
            if "structure_adjacent" == edge_info["type"]:
                r[0] = 1
            elif "semantic_similar_1" == edge_info["type"]:
                r[1] = edge_info["similar_score"]
            elif "same_keyword" == edge_info["type"]:
                r[2] = min(edge_info["same_keyword_count"], 5)
        tokenized_n = tokenizer(neighbour_info["chunk_text"], truncation=True, padding='max_length', max_length=args.text_max_len, add_special_tokens=True, return_tensors='pt')

        n_input_ids = tokenized_n["input_ids"]
        n_attention_mask = tokenized_n["attention_mask"]
        valid_edges = torch.tensor(r, dtype=torch.float32).unsqueeze(0)

        q_input_ids_list.append(q_input_ids)
        q_attention_mask_list.append(q_attention_mask)
        n_input_ids_list.append(n_input_ids)
        n_attention_mask_list.append(n_attention_mask)
        path_input_ids_list.append(path_input_ids)
        path_attention_mask_list.append(path_attention_mask)
        valid_edges_list.append(valid_edges)

    if len(q_input_ids_list) == 0:
        print("**** no more neighbour for select")
        return -1
    
    inputs = {
        "q_input_ids": torch.stack(q_input_ids_list).squeeze(1),
        "q_attention_mask": torch.stack(q_attention_mask_list).squeeze(1), 
        "n_input_ids": torch.stack(n_input_ids_list).squeeze(1), 
        "n_attention_mask": torch.stack(n_attention_mask_list).squeeze(1),
        "path_input_ids": torch.stack(path_input_ids_list).squeeze(1), 
        "path_attention_mask": torch.stack(path_attention_mask_list).squeeze(1),
        "valid_edges": torch.stack(valid_edges_list).squeeze(1)
    }
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}
    with torch.no_grad():
        scores = model(**inputs)
        scores = scores.cpu()
    for name in inputs:
        inputs[name] = inputs[name].cpu()
    del inputs
    torch.cuda.empty_cache()
    scores = scores.squeeze(1).tolist()
    index = scores.index(max(scores))
    return unseen_neighbour_ids[index]

def select_seed_nodes(G, query, query_keyword_list, query_embedding):
    candidate_nodes = {}
    chunk_sum_list = []
    text2nodeid = {}
    for node in G.nodes(data=True):
        try:
            chunk_sum_list.append(node[1]["chunk_text"])
        except:
            print(node)
            print(node[1])
        text2nodeid[node[1]["chunk_text"]] = node[0]
    db = FAISS.from_texts(chunk_sum_list, embeddings)

    similar_docs = db.similarity_search_with_score(query, k=len(chunk_sum_list))
    for doc in similar_docs:
        candidate_nodes[doc[0].page_content]["score"] = doc[1]
    seed_nodes = find_optimal_elements(candidate_nodes, query_keyword_list, similar_docs[-1][1])
    seed_node_ids = [text2nodeid[seed_node] for seed_node in seed_nodes]
    return seed_node_ids



def retriever(args, model, query, graph_pkl_path, retriever_max_step=10):
    query_embedding = embeddings.embed_query(query)
    new_query = query
    for i in range(config.max_extract_keyword_try):
        response = utils.chat_model(utils.extract_query_keyword_template.format(text = new_query))
        try:
            keyword_list = list(json.loads(response))
            break
        except json.JSONDecodeError:
            if i == config.max_extract_keyword_try - 1:
                return  [], "", "extract keyword failed"

    keyword_list = [keyword.replace("'", "").replace('"', "") for keyword in keyword_list ]

    with open(graph_pkl_path, "rb") as graph_f:
        G = pickle.load(graph_f)
    seed_nodes = select_seed_nodes(G, query, keyword_list, query_embedding)
    if len(seed_nodes) == 0:
        return [], {}, "", "empty seed nodes"
    evidence = [G.nodes[seed_node_id]["chunk_text"] for seed_node_id in seed_nodes]
    evidence_each_path = {}
    for index in range(len(seed_nodes)):
        evidence_each_path[index] = [G.nodes[seed_nodes[index]]["chunk_text"]]
    paths = {index:[G.nodes[seed_nodes[index]]["chunk_text"]] for index in range(len(seed_nodes))}
    seen_nodes_each_path = {}
    for index in range(len(seed_nodes)):
        seen_nodes_each_path[index] = [seed_nodes[index]]
    current_nodes = seed_nodes
    next_nodes = []
    for step in range(retriever_max_step):
        for index, node in enumerate(current_nodes):
            best_neighbour_id = select_neighbour(args, model, G, query, node, paths[index], seen_nodes_each_path[index])
            if best_neighbour_id == -1:
                break
            seen_nodes_each_path[index].append(best_neighbour_id)
            evidence_each_path[index].append(G.nodes[best_neighbour_id]["chunk_text"])
            paths[index].append(G.nodes[best_neighbour_id]["chunk_text"])
            next_nodes.append(best_neighbour_id)
        if best_neighbour_id == -1:
                break
        current_nodes = next_nodes
        next_nodes = []
    evidence_list = []
    for path in paths.values():
        evidence_list.append(" ".join(path))
    answer = utils.chat_openai(utils.qa_template.format(question=query, context = " ".join(evidence_list)), "35")
    return evidence, evidence_each_path, answer, ""

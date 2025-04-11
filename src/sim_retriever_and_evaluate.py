import config
import json
import pickle
import math
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import RobertaTokenizer


embeddings = HuggingFaceEmbeddings(model_name="path2model")

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


        inputs = {
            "q_input_ids": q_input_ids,
            "q_attention_mask": q_attention_mask, 
            "n_input_ids": n_input_ids, 
            "n_attention_mask": n_attention_mask,
            "path_input_ids": path_input_ids, 
            "path_attention_mask": path_attention_mask,
            "valid_edges": valid_edges
        }
        scores.append(model(**inputs))
    if len(scores) == 0:
        print("**** no more neighbour for select")
        return -1
    index = scores.index(max(scores))
    return unseen_neighbour_ids[index]

def select_seed_nodes(G, query, mode="key+sim", sim_top_k=0):
    candidate_nodes = {}

    chunk_sum_list = []
    text2nodeid = {}
    for node in G.nodes(data=True):
        node_keywords = [keyword.replace("'", "").replace('"', "") for keyword in node[1]["chunk_keywords"]]

        candidate_nodes[node[1]["chunk_text"]] = {"keywords": node_keywords}
        chunk_sum_list.append(node[1]["chunk_text"])
        text2nodeid[node[1]["chunk_text"]] = node[0]
    db = FAISS.from_texts(chunk_sum_list, embeddings)

    if mode == "sim":
        if sim_top_k == 0:
            raise("sim_top_k should be set if you use sim mode")
        similar_docs = db.similarity_search_with_score(query, k=sim_top_k)
        node_ids = [text2nodeid[doc[0].page_content] for doc in similar_docs]
        return node_ids
    else:
        raise("you can only use sim mode in this python file")



def retriever(query, graph_pkl_path, mode="key+sim", sim_top_k=0):
    with open(graph_pkl_path, "rb") as graph_f:
        G = pickle.load(graph_f)
    seed_nodes = select_seed_nodes(G, query, mode, sim_top_k)
    evidences = [G.nodes[node_id]["chunk_text"] for node_id in seed_nodes]
    return evidences

def get_result(path):
    with open(path, "r") as file:
        lines = file.readlines()
    seed_match_sum = 0
    for line in lines:
        line_info = json.loads(line)
        seed_match_sum += line_info["seed_match"]
    print(seed_match_sum)



dataset = "datset"
data_file_path = "path2data_file"
input_path = "path2input_data"
result_output_path = "path2result_output"
graph_pkl_temp ="path2graph_pkl_dir"


line_start_idx = 0
line_end_idx = 10000
args = config.ARGS
with open(data_file_path, "r") as data_file:
    data_lines = data_file.readlines()
with open(input_path, "r") as original_file:
    original_info = json.loads(original_file.readline())


sim_top_k = 30
for i in range(line_start_idx, line_end_idx):
    result_info = {}
    data = json.loads(data_lines[i])
    pkl_id = data["file_id"] 
    print("============================")
    print("processing ", pkl_id)
    query = original_info[pkl_id]["question"]
    graph_pkl_path = graph_pkl_temp.format(pkl_id=pkl_id, dataset=dataset)
    result_info["query"] = query
    result_info["line_id"] = i
    result_info["pkl_id"] = pkl_id
    result_info["sim_top_k"] = sim_top_k
    correct_evidences = [evidence[1] for evidence in original_info[pkl_id]["supports"]]
    result_info["sim_evidences"] = retriever(query, graph_pkl_path, "sim", sim_top_k)
    seed_match = 0
    for correct_evidence in correct_evidences:
        for evidence in result_info["sim_evidences"]:
            if evidence in correct_evidence or correct_evidence in evidence:
                seed_match += 1
                break
    result_info["seed_match"] = seed_match
    with open(result_output_path, "a") as result_output_file:
        result_output_file.write(json.dumps(result_info) + "\n")
get_result(result_output_path)





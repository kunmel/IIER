import networkx as nx
import json
import config
import pickle
import os
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

model_kwargs = {'device': 'cuda:5'}
embeddings = HuggingFaceEmbeddings(model_name="path2model", model_kwargs=model_kwargs)


def make_vector_db(texts):
    db = FAISS.from_texts(texts, embeddings)
    return db

def has_edge_with_attribute(G, node1, node2, attribute):
    if G.has_edge(node1, node2):
        for edge in G[node1][node2].values():
            if edge["type"] == attribute:
                return True
    return False

def make_graph_pkl(start_idx=0, end_idx=1000,  add_structure_edge=True, add_same_keyword_edge=True, add_similarity_edge=True):
    graph_edge_type = ""
    graph_edge_type += "1" if add_structure_edge else "0"
    graph_edge_type += "1" if add_similarity_edge else "0"
    graph_edge_type += "1" if add_same_keyword_edge else "0"
    graph_edge_type += "rightID"
    edge_sum_count = 0
    graph_sum_count = 0
    node_sum_count = 0
    with open(keyword_path, "r") as keyword_f:
        lines = keyword_f.readlines()[start_idx:end_idx]
        for idx, line in enumerate(tqdm(lines)):
            G = nx.MultiGraph()
            metadata = json.loads(line)
            if len(metadata["chunks"]) == 0:
                continue
            data_id = metadata["data_id"]
            node_id = -1
            text2nodeid = {}
            id2keyword = {}
            text_sum_list = []
            for chunk_metadata in metadata["chunks"]:
                if chunk_metadata["chunk"][1] in text2nodeid.keys():
                    continue
                node_id += 1
                keywords_wo_quote = [keyword.replace("'", "").replace('"', "") for keyword in chunk_metadata["keyword_list"]]
                
                chunk_embedding = embeddings.embed_query(chunk_metadata["chunk"][1])
                G.add_node(node_id, chunk_title=chunk_metadata["chunk"][0], chunk_keywords=chunk_metadata["keyword_list"], chunk_text=chunk_metadata["chunk"][1], chunk_order=chunk_metadata["chunk_order"], chunk_embedding=chunk_embedding, evidence_mark=chunk_metadata["evidence_mark"])
                node_sum_count += 1
                text2nodeid[chunk_metadata["chunk"][1]] = node_id
                text_sum_list.append(chunk_metadata["chunk"][1])
                id2keyword[node_id] = (chunk_metadata["keyword_list"], keywords_wo_quote)
                if add_structure_edge:
                    if chunk_metadata["chunk_order"][1] == 0:
                        continue
                    else:
                        G.add_edge(node_id, node_id-1, type="structure_adjacent")
                        edge_sum_count += 1
                
            if add_similarity_edge:
                vector_db = make_vector_db(text_sum_list)
                for text in text_sum_list:
                    similar_docs = vector_db.similarity_search_with_score(text, k=config.SIMILAR_TOPK_1)
                    for similar_doc in similar_docs:
                        if text2nodeid[text] != text2nodeid[similar_doc[0].page_content] and not has_edge_with_attribute(G, text2nodeid[text], text2nodeid[similar_doc[0].page_content], "semantic_similar_1"):
                            G.add_edge(text2nodeid[text], text2nodeid[similar_doc[0].page_content], type="semantic_similar_1", similar_score=similar_doc[1])
                            edge_sum_count += 1

            if add_same_keyword_edge:
                id_list = list(id2keyword.keys())
                for i in range(len(id_list)):
                    id2same_keyword = {}
                    for j in range(i+1, len(id_list)):
                        same_keywords_wo_quote = set(id2keyword[id_list[i]][1]) & set(id2keyword[id_list[j]][1])
                        same_count = len(same_keywords_wo_quote)
                        if same_count > 0:
                            id2same_keyword[id_list[j]] = (get_keyword_w_quote(same_keywords_wo_quote, id2keyword[id_list[i]][0]), same_count)
                    sorted_d = dict(sorted(id2same_keyword.items(), key=lambda x: x[1][1], reverse=True))
                    for node_id, (same_keywords, same_count) in sorted_d.items():
                        if same_count < config.SAME_KEYWORD_THRESHOLD:
                            break
                        if not has_edge_with_attribute(G, id_list[i], node_id, "same_keyword"):
                            G.add_edge(id_list[i], node_id, type="same_keyword", same_keyword_count=same_count, same_keyword=same_keywords)
                            edge_sum_count += 1
            if not os.path.exists(graph_path_output_prefix+graph_edge_type):
                os.makedirs(graph_path_output_prefix+graph_edge_type)
            with open(os.path.join(graph_path_output_prefix+graph_edge_type, f"{data_id}.pkl"), "wb") as graph_store_f:
                pickle.dump(G, graph_store_f)
                graph_store_f.close()
                graph_sum_count += 1
    print("sum_count: ", graph_sum_count)
    print("edge_sum_count: ", edge_sum_count)
    print("node_sum_count: ", node_sum_count)

def make_data_list(G, data_list, input_info, file_id):
    new_data_list = []
    for idx, (node_id, neighbor_id, path_history) in enumerate(data_list):
        if not G.has_edge(node_id, neighbor_id):
            print("no edge")
            
        edges = dict(G.get_edge_data(node_id, neighbor_id))
        edge_type_list = []
        if node_id == 0:
            continue
        new_data = {"node_id": node_id, "neighbor_id": neighbor_id, "seed_text":G.nodes[node_id]["chunk_text"], "neighbour_text": G.nodes[neighbor_id]["chunk_text"], "query_text": input_info[file_id]["question"]}
        for edge in edges.values():
            edge_type_list.append(edge["type"])
            if edge["type"] == "semantic_similar_1":
                new_data["sim_score"] = float(edge["similar_score"])
            elif edge["type"] == "same_keyword":
                new_data["text_keywords"] = list(edge["same_keyword"])
        new_data["edges"] = edge_type_list
        path_text = ""
        for history_node_id in path_history:
            path_text += G.nodes[history_node_id]["chunk_text"] + " "
        new_data["path_text"] = path_text
        new_data_list.append(new_data)
    return new_data_list


def get_keyword_w_quote(same_keywords, original_keywords):
    keywords_wo_quote = [keyword.replace("'", "").replace('"', "") for keyword in original_keywords]
    keywords_w_quote = [original_keywords[keywords_wo_quote.index(keyword)] for keyword in same_keywords]
    return keywords_w_quote

if __name__ == "__main__":
    edge_type_list = [[True, False, False], [True, True, False], [True, False, True]]
    
    datasets = ["HotpotQA", "Wiki2MQA", "IIRC", "MuSiQue"]
    
    for dataset in datasets:
        for edge_type in edge_type_list:
            keyword_path = "path2keyword"
            graph_path_output_prefix = "path2graph_output"
            make_graph_pkl(0,100, edge_type[0], edge_type[1], edge_type[2])
import pickle
import os
from tqdm import tqdm

path = "path2graph"

density_sum = 0.0
graph_count = 0
semantic_similar_1_edge_per_node = 0
keyword_edge_per_node = 0
struc_edge_per_node = 0
node_sum = 0
edge_sum = 0
for file_name in tqdm(os.listdir(path)):
    file_path = os.path.join(path, file_name)
    with open(file_path, "rb") as graph_file:
        semantic_similar_1_edge_count = 0
        keyword_edge_count = 0
        struc_edge_count = 0
        graph = pickle.load(graph_file)
        node_num = len(graph.nodes)
        node_sum += node_num
        edge_num = len(graph.edges)
        edge_sum += edge_num
        for _, _, edge_data in graph.edges(data=True):
            if edge_data.get('type') == 'semantic_similar_1':
                semantic_similar_1_edge_count += 1
            elif edge_data.get('type') == "same_keyword":
                keyword_edge_count += 1
            elif edge_data.get('type') == "structure_adjacent":
                struc_edge_count += 1
        semantic_similar_1_edge_per_node += semantic_similar_1_edge_count / node_num
        keyword_edge_per_node += keyword_edge_count / node_num
        struc_edge_per_node += struc_edge_count / node_num
        density = 2 * edge_num / (3 * node_num * (node_num - 1))
        density_sum += density
        graph_count += 1
print(density_sum / graph_count)
print(node_sum / graph_count)
print(edge_sum / graph_count)
print(semantic_similar_1_edge_per_node / graph_count)
print(keyword_edge_per_node / graph_count)
print(struc_edge_per_node / graph_count)


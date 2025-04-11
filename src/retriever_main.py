import retriever
import config
import json
import utils
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('model_step', type=str)
parser.add_argument('topk', type=int)
parser.add_argument('max_step', type=int)
parser.add_argument('line_start_idx', type=int)
parser.add_argument('line_end_idx', type=int)
parser.add_argument('--density', type=str)

parser_args = parser.parse_args()


input_path = "path2input_data"
model_path = "path2model"
result_output_path = "path2result_output"
graph_pkl_temp = "path2graph_pkl_dir"


args = config.ARGS
with open(input_path, "r") as original_file:
    original_info = json.loads(original_file.readline())
model = utils.load_model(args, model_path=model_path)


for i in range(parser_args.line_start_idx, parser_args.line_end_idx):
    result_info = {}
    pkl_id = i
    query = original_info[pkl_id]["question"]
    graph_pkl_path = graph_pkl_temp + f"/{pkl_id}.pkl"
    result_info["query"] = query
    result_info["line_id"] = i
    result_info["pkl_id"] = pkl_id

    retriever_evidence, evidence_each_path, llm_answer, error = retriever.retriever(args, model, query, graph_pkl_path, parser_args.max_step)
    if error != "":
        result_info["error"] = error
        with open(result_output_path, "a") as result_output_file:
            result_output_file.write(json.dumps(result_info, ensure_ascii=False) + "\n")
        continue
    correct_evidence = [evidence[1] for evidence in original_info[pkl_id]["supports"]]
    result_info["llm_answer"] = llm_answer
    result_info["correct_answer"] = original_info[pkl_id]["answer"]
    result_info["retriever_evidence"] = retriever_evidence
    result_info["correct_evidence"] = correct_evidence
    result_info["evidence_each_path"] = evidence_each_path

    with open(result_output_path, "a") as result_output_file:
        result_output_file.write(json.dumps(result_info, ensure_ascii=False) + "\n")



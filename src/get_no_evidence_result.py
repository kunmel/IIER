import json
import utils
from tqdm import tqdm


evidence_path_list = ["HotpotQA", "IIRC", "MuSiQue", "Wiki2MQA"]
evidence_path = "path2evidence"
result_output_path = "path2output"


for dataset in evidence_path_list:
    with open(evidence_path.format(dataset=dataset), "r") as evidence_file:
        evidence_lines = json.load(evidence_file)

    for line_info in tqdm(evidence_lines):
        answer = utils.chat_openai(utils.no_evidence_qa_template.format(question=line_info["question"]))
        result = {}
        result["query"] = line_info["question"]
        result["llm_answer"] = answer
        result["correct_answer"] = line_info["answer"]

        with open(result_output_path.format(dataset=dataset), "a") as result_output_file:
            result_output_file.write(json.dumps(result, ensure_ascii=False) + "\n")


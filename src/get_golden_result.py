import json
import utils
from tqdm import tqdm

evidence_path = "path2evidence"
result_output_path = "path2result_output"


with open(evidence_path, "r") as evidence_file:
    evidence_lines = json.load(evidence_file)

for line_info in tqdm(evidence_lines):
    evidence = ""
    for support in line_info["supports"]:
        evidence += support[1]+" "
    
    answer = utils.chat_openai(utils.qa_template.format(question=line_info["question"], context=evidence))
    result = {}
    result["query"] = line_info["question"]
    result["llm_answer"] = answer
    result["correct_answer"] = line_info["answer"]

    with open(result_output_path, "a") as result_output_file:
        result_output_file.write(json.dumps(result, ensure_ascii=False) + "\n")



import logging
import os
import torch
import config
import json
import requests
from model.model import Neighbour_scorer
                                                                                                                                                                     
general_extract_keyword_template = """
Please extract the five most representative keywords from the following text and return them as a list with the keywords separated by commas. 
Example:
John Cecil, 6th Earl of Exeter (15 May 1674 – 24 December 1721), known as Lord Burleigh from 1678 to 1700, was a British peer and Member of Parliament. He was the son of John Cecil, 5th Earl of Exeter, and Anne Cavendish.
Answer: 
["John Cecil, 6th Earl of Exeter", "Lord Burleigh", "British peer", "Member of Parliament", "John Cecil, 5th Earl of Exeter"]

Text:
{text}
Answer:
"""

extract_query_keyword_template = """
Please extract all most representative keywords from the following text and return them as a list with the keywords separated by commas. 
Example:
When did the people who captured Malakoff come to the region where Philipsburg is located?
Answer: 
["Philipsburg", "Malakoff"]
Example:
When was the first establishment that McDonaldization is named after, open in the country Horndean is located?
Answer: 
["McDonaldization", "Horndean"]

Text:
{text}
Answer:
"""

qa_template = """
Given the following question and contexts, create a final answer to the question. 
========= 
QUESTION: {question} 
========= 
CONTEXT: {context}
========= 
QUESTION: {question}  
========= 
ANSWER: Please answer in less than 6 words.
"""

no_evidence_qa_template = """
Given the following question, create a answer to the question. 
========= 
QUESTION: {question} 
========= 
ANSWER: Please answer in less than 6 words.
"""

check_answer_template = """
Given the following question, a correct answer and a candidate answer, please determine if the candidate answer is correct. If the candidate answer covers part or all of the correct answer, answer "yes", otherwise, answer "no". You should only answer "yes" or "no".
QUESTION: {question}
CORRECT ANSWER: {correct_answer}
CANDIDATE ANSWER: {candidate_answer}
"""

check_answer_template_TGP = """You are an expert professor specialized in grading whether the prediction to the question is correct or not according to the real answer.
    ==================
    For example:
    ==================
    Question: What company owns the property of Marvel Comics?
    Answer: The Walt Disney Company
    Prediction: The Walt Disney Company
    Return: 1
    ==================
    Question: Which constituent college of the University of Oxford endows four professorial fellowships for sciences including chemistry and pure mathematics?
    Answer: Magdalen College
    Prediction: Magdalen College.
    Return: 1
    ==================
    Question: Which year was Marvel started?
    Answer: 1939
    Prediction: 1200
    Return: 0
    ==================
    You are grading the following question:
    Question: {question}
    Answer: {correct_answer}
    Prediction: {candidate_answer}
    If the prediction is correct according to answer, return 1. Otherwise, return 0.
    Return: your reply can only be one number '0' or '1'
    """


def load_model(args, model_path=None, ):
    logger.info("Loading model from %s", model_path)
    if not os.path.exists(model_path):
        raise Exception("Model doesn't exists! Train first!")
    state_dict = torch.load(model_path, map_location=config.ARGS.device)
    model = Neighbour_scorer.from_pretrained(config.ARGS.model_name_or_path, config.ARGS)
    model.load_state_dict(state_dict)
    model.to(config.ARGS.device)
    return model

def chat_model(query):
    payload = {
                "model": "model",
                "messages": [{'role': 'user', 'content': query}]}
    response = requests.post("ip", json=payload).json()
    response = response["choices"][0]["message"]["content"]
    return response

def chat_openai(query, mode="35"):
    from openai import AzureOpenAI
    if mode == "35":
        endpoint = "ip"
        deployment = "model"
        client = AzureOpenAI(
        api_key="api_key",
        api_version="api_version",
        azure_endpoint=endpoint
        )
    elif mode == "4":
        endpoint = "ip"
        deployment = "model"
            
        client = AzureOpenAI(
        api_key="api_key",
        api_version="api_version",
        azure_endpoint=endpoint
        )
    else:
        raise Exception("Model size should be 35 or 4 for openai mode")

    for i in range(3):    
        try:
            completion = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                temperature=0.0
            )
        except Exception as e:
            logger.error("Azure Failed")
            return "Azure Failed"
        if json.loads(completion.to_json())["choices"][0]["finish_reason"] == "stop":
            return json.loads(completion.to_json())["choices"][0]["message"]["content"]
    
    return "Azure Failed"

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
logger = logging.getLogger(__name__)
init_logger()

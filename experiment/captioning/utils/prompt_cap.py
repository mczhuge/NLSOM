import os
import openai
import time


# add your OPENAI_API_KEY
openai.api_key = os.getenv('OPENAI_API_KEY')

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

_abc_ = {'0': '(a) ', '1': '(b) ', '2': '(c) ', '3': '(d) ', '4': '(e) '}
_123_ = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3, '(e)': 4,  'a)': 0, 'b)': 1, 'c)': 2, 'd)': 3, 'e)':4, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e' : 4}   

# vlm setting 
_LLM1_ = "InstructGPT"
_VLM1_ = ["BLIP2", "Salesforce/blip2-flan-t5-xl"]
_VLM2_ = ["OFA", 'damo/ofa_visual-question-answering_pretrain_large_en']
_VLM3_ = ["mPLUG", "damo/mplug_visual-question-answering_coco_large_en"]


question_prompt_start = [
                "The first question is ",
                " The second question is ",
                " The third question is ",
                " The fourth question is ",
                " The fifth question is ",
                " The sixth question is ",
                " The seventh question is ",
                " The eighth question is ",
                " The ninth question is ",
                " The tenth question is ",
                "Considering the objective of the first question, now generate another question (end by ?): "
                ]


def prompt_from_VLM(model="ChatGPT", prompt="", max_token=50, role=""):

    if model == "ChatGPT":
        LLM_responce =  openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                        {"role": "system", "content": "You are a {}".format(role)},
                        {"role": "user", "content": prompt},
                        ]
                        )
        answer = LLM_responce["choices"][0]["message"]["content"].strip("\n")

    elif model == "InstructGPT":
        LLM_responce = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=max_token, # 150
                        )
        answer = LLM_responce["choices"][0]["text"].strip("\n")

    elif model == "GPT3":
        LLM_responce = openai.Completion.create(
                        engine='text-ada-001',
                        prompt=prompt,
                        max_tokens=max_token, # 150
                        )
        answer = LLM_responce["choices"][0]["text"].strip("\n")

    else:
        raise NotImplementedError     
       
    return answer, model


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def prompt_from_LLM(model="ChatGPT", prompt="", role="", max_token=100):

    if model == "ChatGPT":
        LLM_responce =  openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                        {"role": "system", "content": "You are a {}".format(role)},
                        {"role": "user", "content": prompt},
                        ]
                        )
        answer = LLM_responce["choices"][0]["message"]["content"].strip("\n")

    elif model == "InstructGPT":
        LLM_responce = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=max_token, # 150
                        )
        answer = LLM_responce["choices"][0]["text"].strip("\n")

    elif model == "GPT3":
        LLM_responce = openai.Completion.create(
                        engine='text-ada-001',
                        prompt=prompt,
                        max_tokens=max_token, # 150
                        )
        answer = LLM_responce["choices"][0]["text"].strip("\n")

    else:
        raise NotImplementedError     
  
    return answer, model



def mission_prompt_generation(task):

    if task == "VQA":
        prologue = "Introduce this image in detail."
    if task == "Image Captioning":
        prologue  = "Introduce this image in detail."

    return prologue 

def mindstorm_prompt_generation(task, question, options, prologue, generated_questions, turn, answer):

    question_collection = ""

    for qid in range(turn+1):
        if qid == 0: 
            question_collection += question_prompt_start[qid] + "\"" +question.strip() + "\"."
        else: 
            question_collection += question_prompt_start[qid] + "\""+ generated_questions[qid].strip() + "\" " +  answer[qid].strip()
        question_collection = question_collection.replace("\n\n", "\n")    

   
    mindstorm_turn_prompt = "There is a {} question: ".format("image captioning")  +  question  + ".\n" +\
                             "This image shows: " + prologue + "Based on these information, we have asked several questions before: " +  question_collection + "\n" + question_prompt_start[-1] 

    return mindstorm_turn_prompt.replace("\n\n", "\n")


def execution_prompt_generation(options, analysis, final_Q, VQA_or_QA):

    hint = analysis.replace("  ", " ")

    execution_prompt  = "There is a {} task: ".format("image captioning") + final_Q + ".\n" + \
                       "The analysis of the image shows: " + hint +  "Consider all informative information. Now organize a frequent and logical description for this image: "

    return execution_prompt.replace("  ", " ").replace("\n\n", "\n")
import os
import openai
import time
import wandb

#from utils.eval import true_of_fault

# add your OPENAI_API_KEY
openai.api_key = os.getenv('OPENAI_API_KEY')

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

_abc_ = {'0': '(a) ', '1': '(b) ', '2': '(c) ', '3': '(d) ', '4': '(e) '}
_123_ = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3, '(e)': 4,  'a)': 0, 'b)': 1, 'c)': 2, 'd)': 3, 'e)':4, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e' : 4}   

question_prompt_start = [
                "The original question is ",
                " The second question is ",
                " The third question is ",
                " The fourth question is ",
                " The fifth question is ",
                " The sixth question is ",
                " The seventh question is ",
                " The eighth question is ",
                " The ninth question is ",
                " The tenth question is ",
                "Considering the options of the original question, now generate another question to help solve the original question (end by ?): "
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
        #mission = "We have a VQA question. First, introduce this image in details."
        prolugue = "Introduce this image in details."
    if task == "Image Captioning":
        #mission = "We have a VQA question. First, introduce this image in details."
        prolugue = "Introduce this image in details."

    return prolugue


def mindstorm_prompt_generation(task, question, options, prologue, generated_questions, turn, answer, mode="monarchical"):

    question_collection = ""

    for qid in range(turn+1):
        if qid == 0: 
            question_collection += question_prompt_start[qid] + "\"" +question.strip() + "\"" + "\nIt has {} options: ".format(len(options)) + str(options).replace("[", "").replace("]", "") + "."
        else: 
            if mode == "monarchical":
                question_collection += question_prompt_start[qid] + "\""+ generated_questions[qid].strip() + "\" " +  answer[qid].strip()
            else:
                question_collection += question_prompt_start[qid] + "\""+ generated_questions[qid].strip() + "\" " +  answer[qid][3].strip()
                
        question_collection = question_collection.replace("\n\n", "\n")    

    
    mindstorm_turn_prompt = "We have a multiple-choice {} task. ".format(task) + " The question is: " +  question  + "\nAnd it has {} options: ".format(len(options)) + str(options).replace("[", "").replace("]", "") + ".\n" +\
                             "The caption of the image is: " + str(prologue).replace("[", "").replace("]", "").strip(".") + ". " + "Based on this information, we have previously asked several questions to other agents and obtained the following answers: " +  question_collection + "\n" + question_prompt_start[-1] 

    return mindstorm_turn_prompt.replace("\n\n", "\n")


def execution_prompt_generation(options, analysis, final_Q, VQA_or_QA):

    hint = analysis.replace("  ", " ")
    if len(options) == 2: 
        execution_prompt  = "We have a multiple-choice VQA task. The question is: {}".format(final_Q) +  "\nAnd It has {} options: ".format(len(options)) + str(options).replace("[", "").replace("]", "")  + ".\n" + \
                            "Context: " + hint +  " Which option do you think is the correct answer? Answer with (a), (b) without explanation." 
    elif len(options) == 3:
        execution_prompt  = "We have a multiple-choice VQA task. The question is: {}".format(final_Q) +"\nAnd It has {} options: ".format(len(options)) + str(options).replace("[", "").replace("]", "") + ".\n"+ \
                            "Context: " + hint + " Which option do you think is the correct answer? Answer with (a), (b), (c) without explanation."
    elif len(options) == 4:
        execution_prompt  = "We have a multiple-choice VQA task. The question is: {}".format(final_Q)  +"\nAnd It has {} options: ".format(len(options)) + str(options).replace("[", "").replace("]", "") + ".\n"+\
                            "Context: " + hint +  " Which option do you think is the correct answer? Answer with (a), (b), (c), or (d) without explanation."
    elif len(options) == 5:
        execution_prompt  = "We have a multiple-choice VQA task. The question is: {}".format(final_Q) +"\nAnd It has {} options: ".format(len(options)) + str(options).replace("[", "").replace("]", "") + ".\n"+\
                            "Context: " + hint + " Which option do you think is the correct answer? Answer with (a), (b), (c), (d) or (e) without explanation." 
    else:
        print('may be error...')     

    return execution_prompt.replace("  ", " ").replace("\n\n", "\n")




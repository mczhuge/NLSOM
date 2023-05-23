import random
import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

def true_of_fault(item, answer_pairs_list, generated_answers, problem_dict, dataset_path, failure_count, options, answer_idx):
    _abc_ = {'0': '(a) ', '1': '(b) ', '2': '(c) ', '3': '(d) ', '4': '(e) '}
    _123_ = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3, '(e)': 4,  'a)': 0, 'b)': 1, 'c)': 2, 'd)': 3, 'e)':4, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e' : 4}  

    if answer_pairs_list[_123_["(a)"]].lower() in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(a)"]]
            
    elif answer_pairs_list[_123_["(b)"]].lower() in generated_answers.lower() :
        generated_answers = answer_pairs_list[_123_["(b)"]]

    elif answer_pairs_list[_123_["(c)"]].lower() in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(c)"]]

    elif answer_pairs_list[_123_["(d)"]].lower() in generated_answers.lower() :
        generated_answers = answer_pairs_list[_123_["(d)"]]

    elif answer_pairs_list[_123_["(e)"]].lower() in generated_answers.lower() : 
        generated_answers = answer_pairs_list[_123_["(e)"]]         

    elif "(a)" in generated_answers.lower() or "a)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(a)"]]

    elif "(b)"  in generated_answers.lower() or "b)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(b)"]]

    elif "(c)" in generated_answers.lower() or "c)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(c)"]]

    elif "(d)" in generated_answers.lower() or "d)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(d)"]]

    elif "(e)" in generated_answers.lower() or "e)" in generated_answers.lower():                                  
        generated_answers = answer_pairs_list[_123_["(e)"]]              
    
    else:
        failure_count += 1
        print("\033[1;31;31mFail to preidct: {}\033[1;m".format(generated_answers))
        random_res = random.randint(0,len(options)-1) 
        generated_answers = answer_pairs_list[random_res]

    print("GT: " + str(answer_pairs_list[int(answer_idx)]) + 'Pred: ' + generated_answers)
    right_or_not = "Right? " + '{' + str(answer_pairs_list[int(answer_idx)] in generated_answers) + '}'
    print("\033[1;31;31m{}\033[1;m".format(right_or_not))    
            
    return generated_answers, problem_dict, failure_count



def transfer_answer(answer_pairs_list, generated_answers, options, lead_answer=None):
    _abc_ = {'0': '(a) ', '1': '(b) ', '2': '(c) ', '3': '(d) ', '4': '(e) '}
    _123_ = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3, '(e)': 4,  'a)': 0, 'b)': 1, 'c)': 2, 'd)': 3, 'e)':4, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e' : 4}  

    if answer_pairs_list[_123_["(a)"]].lower() in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(a)"]]
            
    elif answer_pairs_list[_123_["(b)"]].lower() in generated_answers.lower() :
        generated_answers = answer_pairs_list[_123_["(b)"]]

    elif answer_pairs_list[_123_["(c)"]].lower() in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(c)"]]

    elif answer_pairs_list[_123_["(d)"]].lower() in generated_answers.lower() :
        generated_answers = answer_pairs_list[_123_["(d)"]]

    elif answer_pairs_list[_123_["(e)"]].lower() in generated_answers.lower() : 
        generated_answers = answer_pairs_list[_123_["(e)"]]         

    elif "(a)" in generated_answers.lower() or "a)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(a)"]]

    elif "(b)"  in generated_answers.lower() or "b)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(b)"]]

    elif "(c)" in generated_answers.lower() or "c)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(c)"]]

    elif "(d)" in generated_answers.lower() or "d)" in generated_answers.lower():
        generated_answers = answer_pairs_list[_123_["(d)"]]

    elif "(e)" in generated_answers.lower() or "e)" in generated_answers.lower():                                  
        generated_answers = answer_pairs_list[_123_["(e)"]]              
    
    else:

        print("Wrong answer: ", generated_answers)
        if lead_answer == None:
            random_res = random.randint(0,len(options)-1) 
            generated_answers = answer_pairs_list[random_res]
        else: 
            generated_answers =  lead_answer
        print("Correct wrong answer to: ", generated_answers)

    return generated_answers
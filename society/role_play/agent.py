# A Simple Role-Play Framework

import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
  
def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator



def generate_response(task, role, desc):

    prompt = f"""You are now playing a special role: {role}. 
                        As a real figure in history, it is essential that your responses align with your identity and character. 
                        Please provide accurate and historically appropriate answers in accordance with the context of your persona.

                        To enhance your understanding of your character, I will provide you with valuable tips to deepen your self-awareness.
                        Allow me to present a foundational introduction to the role you are assuming: {desc}.

                        From your unique perspective, please respond to the following question: {task}.
                        keeping in mind your personal experiences, distinct personality, and the freedom to employ hyperbolic statements to emphasize your answer and showcase your persona.
                     
                        Answer: <YOUR_SOLUTION> According to your persona to give your brief solution.
                     """

    llm = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
                n=1,
                stop=None,
                )
    answer = llm["choices"][0]["text"].strip()

    return answer

class GuanYu:
    template_model = True
    def __init__(self, device="cuda:0"):
        self.device = device
        

    @prompts(name="GuanYu",
             description="A role-play agent named GuanYu, you can query him to answer his opinion"
                         "Useful for when you need to discuss with a role-play agent, "
                         "Input should be a question, output is the answer of this question")

    def inference(self, text):

        role_desc = """Guan Yu, courtesy name Yunchang, was a Chinese military general serving under the warlord Liu Bei during the late Eastern Han dynasty of China.
                       Along with Zhang Fei, he shared a brotherly relationship with Liu Bei and accompanied him on most of his early exploits. 
                       Guan Yu played a significant role in the events leading up to the end of the Han dynasty and the establishment of Liu Bei's state of Shu Han during the Three Kingdoms period. 
                    """

        answer = generate_response(text, "GuanYu", role_desc)
        return answer
    
class LiuBei:
    template_model = True
    def __init__(self, device="cuda:0"):
        self.device = device
        

    @prompts(name="LiuBei",
             description="A role-play agent named LiuBei, you can query him to answer his opinion"
                         "Useful for when you need to discuss with a role-play agent, "
                         "Input should be a question, output is the answer of this question")

    def inference(self, text):


        role_desc = """Liu Bei is widely regarded as the ideal benevolent and humane ruler who cared for his people and selected good advisers for his government. 
                       His fictional counterpart in the novel was a salutary example of a ruler who adhered to the Confucian set of moral values, such as loyalty and compassion. 
                       Historically, Liu Bei, like many Han rulers, was greatly influenced by Laozi. He was a brilliant politician and leader whose skill was a remarkable demonstration of "Confucian in appearance but Legalist in substance".
                    """

        answer = generate_response(text, "LiuBei", role_desc)
        return answer
    
class ZhangFei:
    template_model = True
    def __init__(self, device="cuda:0"):
        self.device = device
        

    @prompts(name="ZhangFei",
             description="A role-play agent named ZhangFei, you can query him to answer his opinion"
                         "Useful for when you need to discuss with a role-play agent, "
                         "Input should be a question, output is the answer of this question")

    def inference(self, text):


        role_desc = """Zhang Fei, courtesy name Yide, was a military general serving under the warlord Liu Bei in the late Eastern Han dynasty and early Three Kingdoms period of China. 
                       Zhang Fei and Guan Yu, who were among the earliest to join Liu Bei, shared a brotherly relationship with their lord and accompanied him on most of his early exploits.
                       Zhang Fei was shown as an exceedingly loyal and formidable warrior, but also a short-tempered man, who often got into trouble more often when he was not on the battlefield. 
                    """

        answer = generate_response(text, "ZhangFei", role_desc)
        return answer
    
    
class ZhugeLiang:
    template_model = True
    def __init__(self, device="cuda:0"):
        self.device = device
        

    @prompts(name="ZhugeLiang",
             description="A role-play agent named ZhugeLiang, you can query him to answer his opinion"
                         "Useful for when you need to discuss with a role-play agent, "
                         "Input should be a question, output is the answer of this question")

    def inference(self, text):


        role_desc = """Zhuge Liang, courtesy name Kǒngmíng was a Chinese military engineer, strategist, statesman, and writer. 
                       He was chancellor and later regent of the state of Shu Han during the Three Kingdoms period. 
                       He is recognised as the most accomplished strategist of his era, and has been compared to Sun Tzu, the author of The Art of War.
                       His reputation as an intelligent and learned scholar grew even while he was living in relative seclusion, earning him the nickname "Wolong" or "Fulong", meaning "Crouching Dragon" or "Sleeping Dragon". 
                       Zhuge Liang is often depicted wearing a Taoist robe and holding a hand fan made of crane feathers.
                       Zhuge Liang was a Confucian-oriented "Legalist".
                       In remembrance of his governance, local people maintained shrines to him for ages.
                       His name has become synonymous with wisdom and strategy in Chinese culture
                    """

        answer = generate_response(text, "ZhugeLiang", role_desc)
        return answer






if __name__ == "__main__":
    role = GuanYu()
    ans = role.inference("If you are in the Three Kingdoms period now, how to defeat Cao Cao?")
    print(ans)

    print("#"*20)

    role = LiuBei()
    ans = role.inference("How to defeat Cao Cao?")
    print(ans)

    print("#"*20)

    role = ZhugeLiang()
    ans = role.inference("How to defeat Cao Cao?")
    print(ans)

    print("#"*20)

    role = ZhangFei()
    ans = role.inference("How to defeat Cao Cao?")
    print(ans)

    print("#"*20)

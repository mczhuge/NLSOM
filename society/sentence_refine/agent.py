import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

openai.api_key = os.getenv('OPENAI_API_KEY')
  
def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator
    
class SentenceRefine:
    template_model = True
    def __init__(self, device="cuda:0"):
        self.device = device
        
    @prompts(name="SentenceRefine",
             description="A wrapper around Microsoft bing.com,"
                         "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):
        prompt = "Please revise and slightly expand the following sentence: " + text
        self.instructgpt = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=200,
                    )
        answer = self.instructgpt["choices"][0]["text"].strip("\n")
        return answer

if __name__ == "__main__":
    refine= SentenceRefine()
    ans = refine.inference("AGI is comming.")
    # AGI (Artificial General Intelligence) is rapidly approaching, with many experts predicting that it will be achieved within the next few decades.
    print(ans)
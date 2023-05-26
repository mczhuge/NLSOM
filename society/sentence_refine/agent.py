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
    
class SentenceRefine:
    template_model = True
    def __init__(self, device="cuda:0"):
        self.device = device
        

    @prompts(name="SentenceRefine",
             description="A wrapper around Microsoft bing.com,"
                         "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):
        prompt = "refine this: " + text
        self.instructgpt = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=50,
                    )
        answer = self.instructgpt["choices"][0]["text"].strip("\n")
        return answer

if __name__ == "__main__":
    refine= SentenceRefine()
    ans = refine.inference("Juergen Schmidhuber is a scientist.")
    # Juergen Schmidhuber is a renowned scientist and computer scientist who has made significant contributions to the fields of artificial intelligence and machine learning.
    print(ans)
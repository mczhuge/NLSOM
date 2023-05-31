# Since LangChain is biased toward each tool, we make each search engine anonymous. 
# We will update a version of "Mindstorm" in LangChain, or we write a new NLSOM framework in the future.
# For BingSearch, please refer to https://www.microsoft.com/en-us/bing/apis/bing-web-search-api

import os
import numpy as np

from langchain.tools import Tool
from langchain.utilities import ArxivAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import BingSearchAPIWrapper
# from langchain.utilities import GoogleSearchAPIWrapper
# from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import numpy as np

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class SE_A:
    def __init__(self, device="cpu"):
        self.device = device
        self.wolfram = WolframAlphaAPIWrapper()

    @prompts(name="SE_A",
             #"A wrapper around www.wolfram.org "
             description="A wrapper around XXX.org "
                        "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):

        docs = self.wolfram.run(text)
        return docs.split("\n")[0]

class SE_B:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.arxiv = ArxivAPIWrapper()

    @prompts(name="SE_B",
             # "A wrapper around Arxiv.org "
             description="A wrapper around XXX.org "
                         "Useful for when you need to search information, especially academia information, "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.arxiv.run(text)
        return docs.split("\n\n")[np.random.randint(0,3)]

    
class SE_C:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.wikipedia = WikipediaAPIWrapper()

    @prompts(name="SE_C",
             description="A wrapper around XXX.org "
                         "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.wikipedia.run(text)
        return docs.split("\n\n")[0] #.split("\n")[0:2]
    
class SE_D:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.bing = BingSearchAPIWrapper()

    @prompts(name="SE_D",
             # "A wrapper around Microsoft bing.com,"
             description="A wrapper around XXX.com,"
                         "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.bing.run(text)
        return docs.split("\n")[0:5]


# class DuckDuckGo:
#     def __init__(self, device="cpu"):
#         self.device = "cpu"
#         self.DuckDuckGo = DuckDuckGoSearchRun()

#     @prompts(name="DuckDuckGo",
#              description="A wrapper around search engine DuckDuckGo,"
#                          "Useful for when you need to search information from the internet, "
#                          "Input should be a search query.")
#     def inference(self, text):
#         docs = self.DuckDuckGo.run(text)
#         return docs


if __name__ == "__main__":

    bing = SE_D()
    docs = bing.inference("AGI")

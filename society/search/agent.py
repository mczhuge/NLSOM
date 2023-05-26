#https://www.microsoft.com/en-us/bing/apis/bing-web-search-api


import os
os.environ["BING_SUBSCRIPTION_KEY"] = "62d6585cb6634347ba89f79f84e9313e"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search" #"https://api.cognitive.microsoft.com/bing/v7.0/search" #"https://api.bing.microsoft.com/v7.0/search"
os.environ["WOLFRAM_ALPHA_APPID"] = "QHR6LE-5RRLX85RJT"


from langchain.tools import Tool
from langchain.utilities import ArxivAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import BingSearchAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator
    
class ArxivSearch:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.arxiv = ArxivAPIWrapper()

    @prompts(name="ArxivSearch",
             description="A wrapper around Arxiv.org "
                         "Useful for when you need to search a paper realted to some keywords or author name, "
                         "from scientific articles on arxiv.org. "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.arxiv.run(text)
        return docs
    

class BingSearch:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.bing = BingSearchAPIWrapper()

    @prompts(name="BingSearch",
             description="A wrapper around Microsoft bing.com,"
                         "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.bing.run(text)
        return docs


class DuckDuckGoSearch:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.DuckDuckGo = DuckDuckGoSearchRun()

    @prompts(name="DuckDuckGoSearch",
             description="A wrapper around search engine DuckDuckGo,"
                         "Useful for when you need to search information from the internet, "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.DuckDuckGo.run(text)
        return docs


class WikipediaSearch:
    def __init__(self, device="cpu"):
        self.device = "cpu"
        self.wikipedia = WikipediaAPIWrapper()

    @prompts(name="WikipediaSearch",
             description="A wrapper around www.wikipedia.org "
                         "Useful for when you need to search wikipedia document realted to some keywords or author name, "
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.wikipedia.run(text)
        return docs
    
class WolframAlpha:
    def __init__(self, device="cpu"):
        self.device = device
        self.wolfram = WolframAlphaAPIWrapper()

    @prompts(name="WolframAlpha",
             description="A wrapper around www.wikipedia.org "
                         "Compute expert-level answers using Wolfram’s breakthrough algorithms, knowledgebase and AI technology, "
                         "Useful when you want to calculate the mathmatics problems or want to search some scietific knowledge"
                         "Input should be a search query.")
    def inference(self, text):
        docs = self.wolfram.run(text)
        return docs


if __name__ == "__main__":
    # arxiv = ArxivAPIWrapper()
    # docs = arxiv.inference("Juergen Schmidhuber")

    # wiki =  WikipediaSearch()
    # docs = wiki.inference("Juergen Schmidhuber")

    # wolframalpha = WolframAlpha()
    # docs = wolframalpha.inference("Who is Juergen Schmidhuber?")

    bing = BingSearch()
    docs = bing.inference("Juergen Schmidhuber")

    # ddgo = DuckDuckGoSearch()
    # docs = ddgo.inference("Juergen Schmidhuber")



    print(docs)

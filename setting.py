from pathlib import Path

APP_NAME = "NLSOM"
MODEL = "gpt-3.5-turbo"
PAGE_ICON = "🤯"

CHUNK_SIZE = 1000

DATA_PATH = Path.cwd() / "data"
REPO_URL = "https://github.com/AI-Initiative-KAUST/NLSOM"

AUTHENTICATION_HELP = f"""
The keys are neither exposed nor made visible or stored permanently in any way.\n
"""

USAGE_HELP = f"""
These are the accumulated OpenAI API usage metrics.\n
The app uses '{MODEL}' for chat and 'text-davinci-003' for recommendations or role-players.\n
Learn more about OpenAI's pricing [here](https://openai.com/pricing#language-models)
"""

OPENAI_HELP = """
You can sign-up for OpenAI's API [here](https://openai.com/blog/openai-api).\n
Once you are logged in, you find the API keys [here](https://platform.openai.com/account/api-keys)
"""

HUGGINGFACE_HELP = """
For Huggingface, please refer to https://huggingface.co/inference-api
"""

BINGSEARCH_HELP = """
For BingSearch, please refer to https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
"""

WOLFRAMALPHA_HELP = """
Please refer to https://products.wolframalpha.com/api
"""

REPLICATE_HELP = """
Please refer to https://replicate.com
"""

MODELSCOPE_HELP = """
Please refer to http://www.modelscope.cn
"""

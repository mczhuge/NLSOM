import logging
import os
import re
import shutil
import sys
from typing import List

#import deeplake
import openai
import streamlit as st
#from dotenv import load_dotenv
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.llms.openai import OpenAI


from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake, VectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import (
    APP_NAME,
    CHUNK_SIZE,
    DATA_PATH,
    PAGE_ICON,
    REPO_URL,
    TEMPERATURE,
    K,
)

from bak.prompt import (
    NLSOM_PREFIX, 
    NLSOM_FORMAT_INSTRUCTIONS, 
    NLSOM_SUFFIX, 
)

# OpenAI Agent
nlsom_organizer = OpenAI(temperature=0)
nlsom_memory = ConversationBufferMemory(memory_key="chat_history", output_key="output")

# loads environment variables
# load_dotenv()

logger = logging.getLogger(APP_NAME)


def configure_logger(debug: int = 0) -> None:
    # boilerplate code to enable logging in the streamlit app console
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logger.setLevel(log_level)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter("%(message)s")

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False


configure_logger(0)


def authenticate(
    #openai_api_key: str, activeloop_token: str, activeloop_org_name: str
    openai_api_key: str
) -> None:
    # Validate all credentials are set and correct
    # Check for env variables to enable local dev and deployments with shared credentials
    openai_api_key = (
        openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY")
    )
    # activeloop_token = (
    #     activeloop_token
    #     or os.environ.get("ACTIVELOOP_TOKEN")
    #     or st.secrets.get("ACTIVELOOP_TOKEN")
    # )
    # activeloop_org_name = (
    #     activeloop_org_name
    #     or os.environ.get("ACTIVELOOP_ORG_NAME")
    #     or st.secrets.get("ACTIVELOOP_ORG_NAME")
    # )
    # if not (openai_api_key and activeloop_token and activeloop_org_name):
    if not (openai_api_key):
        st.session_state["auth_ok"] = False
        st.error("Credentials neither set nor stored", icon=PAGE_ICON)
        return
    try:
        # Try to access openai and deeplake
        with st.spinner("Authentifying..."):
            openai.api_key = openai_api_key
            openai.Model.list()
            # deeplake.exists(
            #     f"hub://{activeloop_org_name}/DataChad-Authentication-Check",
            #     token=activeloop_token,
            # )
    except Exception as e:
        logger.error(f"Authentication failed with {e}")
        st.session_state["auth_ok"] = False
        st.error("Authentication failed", icon=PAGE_ICON)
        return
    # store credentials in the session state
    st.session_state["auth_ok"] = True
    st.session_state["openai_api_key"] = openai_api_key
    # st.session_state["activeloop_token"] = activeloop_token
    # st.session_state["activeloop_org_name"] = activeloop_org_name
    logger.info("Authentification successful!")


def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    # streamlit uploaded files need to be stored locally
    # before embedded and uploaded to the hub
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    file_path = str(DATA_PATH / uploaded_file.name)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    file = open(file_path, "wb")
    file.write(file_bytes)
    file.close()
    logger.info(f"Saved: {file_path}")
    return file_path


def delete_uploaded_file(uploaded_file: UploadedFile) -> None:
    # cleanup locally stored files
    file_path = DATA_PATH / uploaded_file.name
    if os.path.exists(DATA_PATH):
        os.remove(file_path)
        logger.info(f"Removed: {file_path}")


def handle_load_error(e: str = None) -> None:
    error_msg = f"Failed to load '{st.session_state['data_source']}':\n\n{e}"
    st.error(error_msg, icon=PAGE_ICON)
    logger.error(error_msg)
    st.stop()


def load_git(data_source: str, chunk_size: int = CHUNK_SIZE) -> List[Document]:
    # We need to try both common main branches
    # Thank you github for the "master" to "main" switch
    # we need to make sure the data path exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    repo_name = data_source.split("/")[-1].split(".")[0]
    repo_path = str(DATA_PATH / repo_name)
    clone_url = data_source
    if os.path.exists(repo_path):
        clone_url = None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    branches = ["main", "master"]
    for branch in branches:
        try:
            docs = GitLoader(repo_path, clone_url, branch).load_and_split(text_splitter)
            break
        except Exception as e:
            logger.error(f"Error loading git: {e}")
    if os.path.exists(repo_path):
        # cleanup repo afterwards
        shutil.rmtree(repo_path)
    try:
        return docs
    except:
        msg = "Make sure to use HTTPS git repo links"
        handle_load_error(msg)


def load_any_data_source(
    data_source: str, chunk_size: int = CHUNK_SIZE
) -> List[Document]:
    # Ugly thing that decides how to load data
    # It aint much, but it's honest work
    is_img = data_source.endswith(".png")
    is_video = data_source.endswith(".mp4")
    is_audio = data_source.endswith(".wav")
    is_text = data_source.endswith(".txt")
    is_web = data_source.startswith("http")
    is_pdf = data_source.endswith(".pdf")
    is_csv = data_source.endswith("csv")
    is_html = data_source.endswith(".html")
    is_git = data_source.endswith(".git")
    is_notebook = data_source.endswith(".ipynb")
    is_doc = data_source.endswith(".doc")
    is_py = data_source.endswith(".py")
    is_dir = os.path.isdir(data_source)
    is_file = os.path.isfile(data_source)

    loader = None
    if is_dir:
        loader = DirectoryLoader(data_source, recursive=True, silent_errors=True)
    elif is_git:
        return load_git(data_source, chunk_size)
    elif is_web:
        if is_pdf:
            loader = OnlinePDFLoader(data_source)
        else:
            loader = WebBaseLoader(data_source)
    elif is_file:
        if is_text:
            loader = TextLoader(data_source, encoding="utf-8")
        elif is_notebook:
            loader = NotebookLoader(data_source)
        elif is_pdf:
            loader = UnstructuredPDFLoader(data_source)
        elif is_html:
            loader = UnstructuredHTMLLoader(data_source)
        elif is_doc:
            loader = UnstructuredWordDocumentLoader(data_source)
        elif is_csv:
            loader = CSVLoader(data_source, encoding="utf-8")
        elif is_py:
            loader = PythonLoader(data_source)
        else:
            loader = UnstructuredFileLoader(data_source)
    try:
        # Chunk size is a major trade-off parameter to control result accuracy over computaion
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        docs = loader.load_and_split(text_splitter)
        logger.info(f"Loaded: {len(docs)} document chucks")
        return docs
    except Exception as e:
        msg = (
            e
            if loader
            else f"No Loader found for your data source. Consider contributing:  {REPO_URL}!"
        )
        handle_load_error(msg)


def update_usage(cb: OpenAICallbackHandler) -> None:
    # Accumulate API call usage via callbacks
    logger.info(f"Usage: {cb}")
    callback_properties = [
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_cost",
    ]
    for prop in callback_properties:
        value = getattr(cb, prop, 0)
        st.session_state["usage"].setdefault(prop, 0)
        st.session_state["usage"][prop] += value


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    
    tokens = str(history_memory).replace("[(", "").replace(")]", "").split()
    n_tokens = len(tokens)
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

def generate_response(prompt: str, tools, history) -> str:

    mindstorm = initialize_agent(
        tools,
        nlsom_organizer,
        agent="conversational-react-description",
        verbose=True,
        memory=nlsom_memory,
        return_intermediate_steps=True,
        agent_kwargs={'prefix': NLSOM_PREFIX, 'format_instructions': NLSOM_FORMAT_INSTRUCTIONS,
                    'suffix': NLSOM_SUFFIX}, )

    mindstorm.memory.chat_memory.add_user_message(st.session_state["chat_history"][0][0])
    mindstorm.memory.chat_memory.add_user_message(st.session_state["chat_history"][0][1])

    response = mindstorm({'input': prompt.strip()})
    response['output'] = response['output'].replace("\\", "/")
    response = re.sub('(data/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', response['output'])

    logger.info(f"Response: '{response}'")
    st.session_state["chat_history"].append((prompt, response))
    return response
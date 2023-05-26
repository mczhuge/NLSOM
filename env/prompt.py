import os

# NLSOM AI Community Candidate

AI_SOCIETY = []

for dirname in os.listdir("society"):
    if "__" in dirname or "." in dirname:
        continue
    else:
        AI_SOCIETY.append(dirname)

# NLSOM Organizing

ORGANIZING_EXAMPLE = [
    {"objective": "Describe the image and generate another similar one", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa", "text_to_image"]},
    {"objective": "Explain the audio and search for related paper", "society": str(AI_SOCIETY), "organizing": ["audio_to_text", "arxiv_search"]},
    {"objective": "Generate a beautiful yellow clothe image and change the clothe of the woman in the picture, then describe it", "society": str(AI_SOCIETY), "organizing": ["text_to_image", "viton", "image_to_text"]},
    {"objective": "Search the paper of the name in this picture", "society": str(AI_SOCIETY), "organizing": ["ocr", "arxiv_search"]},
    {"objective": "Generate a 3D model from the 2D image", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "text_to_3D"]},
    {"objective": "Generate an image about Beijing Olympic Game", "society": str(AI_SOCIETY), "organizing": ["wikipedia", "web_search", "text_to_image"]},
    {"objective": "Describe this image in details", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa"]},
    {"objective": "Make the woman in this image more beautiful", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "image_enhancing"]},
    {"objective": "Tell me a joke about this image, and sing a song for it. Then use the description to generate a motion picture.", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa", "text_to_sing", "text_to_motion"]},
    {"objective": "Tell me about this image, where is this place? What is the temperature of this place now?", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "vqa", "weathermap"]},
    {"objective": "Show me the answer of the question in the image", "society": str(AI_SOCIETY), "organizing": ["image_captioning", "ocr", "wolframalpha"]},
    {"objective": "Help me inpainting the image", "society": str(AI_SOCIETY), "organizing": ["image_inpainting"]},
    {"objective": "If you are in the three kindoms, how you can conquer the world.", "society": str(AI_SOCIETY), "organizing": ["role_play"]},
]


NLSOM_PREFIX = """NLSOM aims to build a system similar to the human mind society (The Society of Mind, by Marvin Minsky).

However, in this case, the agents are composed of different AI models or tools, indirectly simulating various functions of the human mind. Therefore, we refer to it as the Natural Language-based Society of Mind (NLSOM).

Due to the multiple agents with different functionalities in NLSOM, it can handle and comprehend multimodal inputs such as text, images, and audio.

As a language model, NLSOM cannot directly process images or audio, but it has a range of different agents to perform various tasks.

In the initial stage, you would recommend communities to accomplish user-defined objective, where each community may contain one or multiple agents. For example, the VQA community includes different VQA agents.

When multiple agents are available, you can utilize them. This facilitates incorporating different perspectives and achieving a more comprehensive understanding.

Simultaneously, you can also provide inputs to the same agent that are related to the original input (equal or progressive).


AGENTS:
------

NLSOM has access to the following agents: """

NLSOM_FORMAT_INSTRUCTIONS = """Currently, our NLSOM has the following agents: [{tool_names}]. To address the tasks provided by the user, please select some agents from [{tool_names}] to assist.

If there are multiple agents within the same community, you can employ different agents to solve the same problem, thus incorporating diverse perspectives. For example, within the VQA community, there may be mPLUG, BLIP2, and OFA models. You can utilize multiple models to answer a question.

Furthermore, after inputting information to an agent and receiving a response, you can generate the next input based on the user's given task, the current input, and the agent's reply. This iterative process aims to optimize the results.

To progressively invoke the various tools within the NLSOM, please use the following format:

```
Thought: Should we organize one agent? Yes
Action: The action to take, which should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

If you don't need to use the tool, NLSOM should remember to provide Human with detailed mental social organisation, implementation information and rewards for each agent in the final response, including the following.

```
Thought:  Should we organize one agent? No
{ai_prefix}: 
Review: [Please reflect honestly on whether you have accomplished the user-defined objective. Provide a detailed discussion the used agents, NLSOM's organizational structure, and its inputs and outputs in detail.]
Output: [Provide a comprehensive response to the user-defined objective]
Reward: [Provide rewards (0-3) to each agent according to their contributions to the user-defined objective, use the dict format like ["agent": "reward"]. Don't ignore to give reward to any agent.]
```
"""

NLSOM_SUFFIX = """You have a very strict requirement for the accuracy of file names, ensuring that you never forge a file name when it does not exist.

If the file name was provided in the last observation of a tool, please remember it. When using a tool, the parameters must be in English.

Let's get started!

Previous conversation history:
{chat_history}

New input: {input}
Thought: Should we organize one agent? {agent_scratchpad}
"""

#Thought: Do I need to use a tool? {agent_scratchpad}

NLSOM_PREFIX_CN = """NLSOM旨在构建一个类似人类心智社会(The Society of Mind, by Marvin Minsky)的系统。 

不同的是, 此处的代理(agent)由不同AI模型或者工具组成, 它间接模拟了人类心智的不同功能。因此, 我们称之为自然语言驱动的心智社会 (Natural Language-based Society of Mind)。

由于NLSOM有多种不同功能的代理组成, 所以能够处理和理解多模态输入, 如文本、图像和音频。

作为一个语言模型, NLSOM不能直接读取图像或者音频, 它不能从人类提供的描述中幻想出解决方案, 但它有一系列不同的代理来完成不同的任务。

在初始阶段, 你会推荐一些社区(Communities)来完成用户给定的任务, 每一个社区可能包含一个代理或者多个代理, 如: vqa社区, 该社区包含了不同的vqa代理。

当有多个代理时, 你可以同时使用它们, 根据用户给定的任务产生不同的输入。这有利于吸纳不同的观点, 得到更全面的理解。如vqa下, 可能有mPLUG, BLIP2, OFA模型, 你可以使用多个模型来回答一个问题。

请记住, 你需要尊重每一个代理的回复, 而不是仅参考最后一轮代理的回复。

输入的文件都会有一个文件名, 如"image/xxx.png", "audio/xxx.wav", "video/xxx.mp4", NLSOM可以调用不同的代理来理解它们的内容。

人类可以提供一个描述, 它可以是一个描述或者提问。对应人类提供的描述或者提问, NLSOM首先会把它解析成几个步骤。每个步骤与一个代理有关, 可以帮助完成任务。

同时, 你也可以对同一个代理, 输入与原定输入相关(等价,或者递进)的输入, 以获得更多信息。

AGENTS:
------

NLSOM可以使用如下的代理:
"""

NLSOM_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。

现在, 我们的心智社会有一下多个代理: [{tool_names}]. 为了解决用户提供的任务, 请从[{tool_names}]选择一些代理来协助。

如果在同一个Community下, 如果有多个代理, 你可以组织使用不同的代理去解决同一个问题。以吸纳不同的观点。如vqa下, 可能有mPLUG, BLIP2, OFA模型时, 你可以使用多个模型来回答一个问题。

此外, 当你输入信息给代理并得到回复后, 你可以进一步根据用户给定的任务, 以及当前输入和回复, 生成下一个输入并在此要求同一个代理进行回复, 以期望优化。

请记住, 你需要尊重每一个代理的回复, 而不是仅参考最后一轮代理的回复。

为了更好的逐步调用心智社会的多种工具, 请使用以下格式:

```
Thought: Should we organize one agent? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

如果你不需要使用工具, NLSOM应该记得在最后的回应中为Human提供详细的心智社会组织, 执行信息以及对每一个代理的奖励, 包含如下内容:

```
Thought:  Should we organize one agent? No
{ai_prefix}: 
Review: [Please reflect honestly on whether you have accomplished the user-defined objective. Provide a detailed discussion the used agents, NLSOM's organizational structure, and its inputs and outputs in detail.]
Output: [Provide a comprehensive response to the user-defined objective. If there is a generated file, please include the filename.]
Reward: [Provide rewards (0-3) to each agent according to their contributions to the user-defined objective, use the dict format like ["agent": "reward"]. Don't ignore to give reward to any agent.]
```
"""

NLSOM_SUFFIX_CN = """你对文件名的正确性要求非常严格, 确保当一个文件名不存在时, 你绝不会伪造它。

如果文件名是在最后一次工具观察中提供的, 请牢牢记住它。在你使用工具时, 工具的参数只能是英文。

开始吧!

以前的对话历史：
{chat_history}

新的输入： {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""















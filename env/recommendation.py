import os
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI

from env.prompt import AI_SOCIETY, ORGANIZING_EXAMPLE

in_context_template = """Objective: {objective}\nSociety: {society}\nOrganizing: {organizing}""" 

example_prompt = PromptTemplate(
                    input_variables=["objective", "society", "organizing"],
                    template=in_context_template,
                )

few_shot_prompt = FewShotPromptTemplate(
                examples=ORGANIZING_EXAMPLE,
                example_prompt=example_prompt,
                prefix="Organize the AI society",
                suffix="Objective: {input}\nSociety: {society}\nOrganizing: ",
                input_variables=["input", "society"],
                #example_separator="\n\n",
            )

def Organize(objective):
    NLSOM_candiate = few_shot_prompt.format(input=objective, society=str(AI_SOCIETY))
    llm = OpenAI(temperature=0.1)
    return llm(NLSOM_candiate)

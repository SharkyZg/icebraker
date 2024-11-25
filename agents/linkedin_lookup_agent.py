import os
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import(
    create_react_agent,
    AgentExecutor
)
from langchain import hub

def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    template = "given the full name {name_of_person} I want you to get bac$k the LinkedIn Page URL. Your answer should contain only a URL."

    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"])

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 LinkedIn profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the LinkedIn Page URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, prompt=react_prompt, tools=tools_for_agent)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    return agent_executor.invoke({"input": prompt_template.format_prompt(name_of_person=name)})

if __name__ == "__main__":
    linkedin_url = lookup("Eden Marco Udemy")
    print(linkedin_url)
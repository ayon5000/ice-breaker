import os
import sys
from dotenv import load_dotenv

_ = load_dotenv()

python_path = os.getenv("PYTHONPATH")
if python_path:
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), python_path))
    if full_path not in sys.path:
        sys.path.append(full_path)

from tools.tools import get_profile_url_tavily
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub


def lookup(name: str) -> str:

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    template = """
    Given the full name {name_of_person}, I want you to find a link to their Twitter profile page, and extract from it their username.
    In Your Final answer only the person's username.
    """

    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url_tavily,
            description="useful for when you need to get the Twitter page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    twitter_username = result["output"]

    return twitter_username


if __name__ == "__main__":
    twitter_username = lookup(name="Eden Marco Udemy")
    print(twitter_username)

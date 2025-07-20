from typing import Tuple
from dotenv import load_dotenv

_ = load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from output_parsers import summary_parser, Summary


def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username,
        mock=True,
    )

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(
        username=twitter_username,
        mock=True,
    )

    summary_template = """
    Given the information about a person from LinkedIn {information},
    and their latest Twitter posts {twitter_posts} I want you to create:
    1. A short summary
    2. two interesting facts about them 

    Use information from both Twitter and Linkedin.
    \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")

    # chain = summary_prompt_template | llm | StrOutputParser()
    chain = summary_prompt_template | llm | summary_parser

    res: Summary = chain.invoke(
        input={"information": linkedin_data, "twitter_posts": tweets}
    )

    return res, linkedin_data["photoUrl"]


if __name__ == "__main__":
    print("Hello Langchain")
    ice_break_with(name="Eden Marco Udemy")

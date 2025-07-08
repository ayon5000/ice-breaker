from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile

_ = load_dotenv()

if __name__ == "__main__":
    print("Hello Langchain")

    summary_template = """
    given the LinkedIn information {information} about a person, I want you to create:
    1. A short summary
    2. two interesting facts about them 
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")

    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/",
        mock=True,
    )

    res = chain.invoke(input={"information": linkedin_data})

    print(res)

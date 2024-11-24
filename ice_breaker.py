from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv_vault import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello, World!")
    print(os.getenv("OPENAI_API_KEY"))

    summary_template = """
    Given the linkedin information {information} about a person, I want you to create:
    1. A short summary of the person
    2. Two interesting facts about the person
    """

    summary_prompt = PromptTemplate.from_template(summary_template)

    summary_chain = summary_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco", mock=True)
    response = summary_chain.invoke({"information": linkedin_data})
    print(response.content)

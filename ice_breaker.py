from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv_vault import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser, Summary
from typing import Tuple
def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin['output'], mock=True)

    summary_template = """
    Given the linkedin information {information} about a person, I want you to create:
    1. A short summary of the person
    2. Two interesting facts about the person

   \n{format_instructions}
    """

    summary_prompt = PromptTemplate.from_template(summary_template, partial_variables={"format_instructions": summary_parser.get_format_instructions()})

    summary_chain = summary_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | summary_parser
    response:Summary = summary_chain.invoke({"information": linkedin_data})
    return response, linkedin_data.get("profile_pic_url")

if __name__ == "__main__":
    load_dotenv()
    print("Ice Breaker")
    ice_break_with("Marko Sarkanj")

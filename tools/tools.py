from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(name: str):
    search = TavilySearchResults()
    return search.run(f"{name} LinkedIn profile")
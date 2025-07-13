# https://docs.tavily.com/documentation/integrations/langchain#tavily-search
from langchain_tavily import TavilySearch


def get_profile_url_tavily(name: str):
    """Searches for LinkedIn or Twitter profile page"""
    search = TavilySearch(
        max_results=5,
        topic="general",
        # include_answer=False,
        # include_raw_content=False,
        # include_images=False,
        # include_image_descriptions=False,
        # search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )

    res = search.invoke({"query": f"{name}"})

    return res

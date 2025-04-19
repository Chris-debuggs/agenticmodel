from langchain_community.tools import WikipediaQueryRun , DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func=search.run,
    description="A tool to search the web. Use this tool to get information from the web.",
    return_direct=False,
)

api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max =1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
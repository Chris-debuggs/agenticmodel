from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ai21 import ChatAI21
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from tools import search_tool,wiki_tool
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize the LLM and parser
llm = ChatAI21(model="jamba-instruct")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         You are a research assistant that will help generate a research paper
         Answer the query and use necessary tools
         Wrap the output in this format and provide no other text\n{format_instructions}
         """,
         ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create agent and executor
tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
query = input("Enter your research query: ")
# Test the setup
try:
    raw_response = agent_executor.invoke({
        "query": query,
        "chat_history": []
    })
    print("Raw response:", raw_response)
    
    if isinstance(raw_response.get("output"), str):
        structured_response = parser.parse(raw_response["output"])
        print("\nStructured response:", structured_response)
    else:
        print("\nUnexpected response format:", raw_response)
except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Raw response received:", raw_response)

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from ddgs import DDGS


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

MAX_SEARCH_RESULTS = 2  # ✅ Limit to 3 sites

def run_ddg_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query)
        results_list = list(results)[:MAX_SEARCH_RESULTS]  # ✅ Take only first N results
        return "\n".join(f"{res['title']}: {res['href']}" for res in results_list)


# def run_ddg_search(query: str) -> str:
#     with DDGS() as ddgs:
#         results = ddgs.text(query)
#         return "\n".join(f"{res['title']}: {res['href']}" for res in results[:5])


search_tool = Tool(
    name="search",
    func=run_ddg_search,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
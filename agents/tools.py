import os
import logging
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urlparse

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from ddgs import DDGS

# --- setup logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- SAVE TOOL ---
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted)
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file."
)

# --- DUCKDUCKGO SEARCH ---
MAX_SEARCH_RESULTS = 3
def run_ddg_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Search with DuckDuckGo and return titles + URLs."""
    with DDGS() as ddgs:
        items = list(ddgs.text(query))[:max_results]
        return "\n".join(f"{r['title']}: {r['href']}" for r in items)

ddg_search_tool = Tool(
    name="duckduckgo_search",
    func=run_ddg_search,
    description="Searches the web using DuckDuckGo and returns titles and URLs."
)

# --- GOOGLE CUSTOM SEARCH ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in environment")

def google_search(query: str, max_results: int = 5) -> str:
    """
    Use Google Custom Search JSON API to fetch titles and URLs.
    Falls back to DuckDuckGo on any HTTP error.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx":  GOOGLE_CSE_ID,
        "q":   query,
        "num": max_results
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return "\n".join(f"{i['title']}: {i['link']}" for i in items)
    except requests.RequestException as e:
        logger.warning(f"Google Search failed ({e}), falling back to DuckDuckGo")
        return run_ddg_search(query, max_results=max_results)

google_search_tool = Tool(
    name="google_search",
    func=google_search,
    description="Search the web via Google Custom Search and return titles and URLs."
)

# --- GENERIC URL SCRAPER ---
def scrape_url(url: str) -> str:
    """Fetch a URL and return its paragraph text, with a browser-like User-Agent."""
    resp = requests.get(
        url,
        timeout=5,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

scrape_tool = Tool(
    name="scrape_url",
    func=scrape_url,
    description="Scrape a web page and return its text content (paragraphs only)."
)

# --- ARTICLE EXTRACTION ---
def extract_article(url: str) -> dict:
    """
    Download and parse an article, returning its title, authors, date, and summary.
    """
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return {
        "title": article.title,
        "authors": article.authors,
        "publish_date": article.publish_date.isoformat() if article.publish_date else None,
        "summary": article.summary,
    }

extract_tool = Tool(
    name="extract_article",
    func=extract_article,
    description="Given an article URL, extract and return its title, authors, date, and summary."
)

# --- WIKIPEDIA TOOL ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

__all__ = [
    "save_to_txt", "save_tool",
    "run_ddg_search", "ddg_search_tool",
    "google_search", "google_search_tool",
    "scrape_url", "scrape_tool",
    "extract_article", "extract_tool",
    "wiki_tool"
]

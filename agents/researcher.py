import os, re, json, time, logging
from typing import List, Dict, Any, Optional
import concurrent.futures
import functools
import google.generativeai as genai
from urllib.parse import urlparse
from dotenv import load_dotenv

from .tools import (
    run_ddg_search,
    google_search,
    wiki_tool,
    scrape_url,
    extract_article,
    save_to_txt
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")
genai.configure(api_key=GEMINI_API_KEY)

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# configurable models & temps
MODELS = {
    'local': ('gemini-2.0-flash', 0.0),
    'world': ('gemini-2.0-flash', 0.0),
    'web':   ('gemini-2.5-pro',   0.0),
    'merge': ('gemini-2.5-pro',   0.2),
}

@functools.lru_cache(maxsize=128)
def _cached_scrape(url: str) -> str:
    return scrape_url(url)

def retry(max_attempts=3, delay: float = 1.0):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed for {func.__name__}: {e}")
                    time.sleep(delay)
            return ""
        return wrapper
    return deco

scrape_with_retry = retry()(_cached_scrape)

def _call_llm(stage: str, prompt: str) -> Any:
    model_name, temp = MODELS.get(stage, MODELS['world'])
    llm = genai.GenerativeModel(model_name)
    prompt += "\n\nInstructions: Append inline citations like (Source: ...) where relevant."
    resp = llm.generate_content(prompt, generation_config={"temperature": temp})
    text = resp.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": text}

def is_valid_url(u: str) -> bool:
    p = urlparse(u)
    return p.scheme in ("http", "https") and bool(p.netloc)

class ResearchAgent:
    def __init__(self, indexer):
        self.indexer = indexer

    def generate_report(self, query: str, mode: str = "normal") -> Dict[str, Any]:
        start_time = time.time()
        mode = (mode or "normal").lower()
        logger.info(f"Generating report for '{query}' in {mode} mode")

        # override via flags
        if query.startswith("wiki:"):
            mode = "deep"
            query = query[len("wiki:"):].strip()
        elif query.startswith("news:"):
            mode = "deep"
            query = query[len("news:"):].strip()

        # 1) LOCAL context
        raw_results = self.indexer.search(query) or []
        chunks: List[str] = []
        for item in raw_results:
            if isinstance(item, dict):
                text, score = item.get('text',''), item.get('score',0)
            elif isinstance(item, tuple):
                text, score = item[0], item[1]
            else:
                text, score = str(item), 1.0
            if score < 0.1:
                break
            chunks.append(text)
        local_text = "\n".join(chunks)

        local_prompt = json.dumps({
            "source": "local",
            "query": query,
            "context": local_text,
            "instructions": "Answer only from LOCAL context, extremely concise. If not found, return {\"answer\": null}. No filler."
        })
        local_future = concurrent.futures.ThreadPoolExecutor().submit(
            _call_llm, 'local', local_prompt
        )

        # 2) WORLD or DEEP
        if mode == "normal":
            world_prompt = json.dumps({
                "source": "world",
                "query": query,
                "instructions": "Use ONLY your internal knowledge, be extremely concise, and always provide an answer—even if it wasn’t in your LOCAL context."
            })
            other_future = concurrent.futures.ThreadPoolExecutor().submit(
                _call_llm, 'world', world_prompt
            )
            web_urls: List[str] = []
        else:
            raw_res = google_search(query, max_results=5) or ""
            if not raw_res.strip():
                raw_res = run_ddg_search(query, max_results=5)
            lines = raw_res.splitlines()

            web_urls: List[str] = []
            for line in lines:
                if ": http" not in line:
                    continue
                _, candidate = line.rsplit(": ", 1)
                if is_valid_url(candidate):
                    web_urls.append(candidate)
                if len(web_urls) >= 5:
                    break

            texts, sources = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
                futures = {exe.submit(scrape_with_retry, url): url for url in web_urls}
                for fut in concurrent.futures.as_completed(futures):
                    url = futures[fut]
                    txt = fut.result()
                    if not txt:
                        continue
                    try:
                        art = extract_article(url)
                        sources.append({
                            "name": art['title'],
                            "url": url,
                            "meta": {
                                "authors": art['authors'],
                                "date": art['publish_date']
                            }
                        })
                        texts.append(art['summary'])
                    except Exception:
                        sources.append({"name": url, "url": url})
                        texts.append(txt)

            wiki_snip = wiki_tool.run(query) or ""
            web_text = "\n".join([wiki_snip] + texts)
            web_prompt = json.dumps({
                "source": "web",
                "query": query,
                "instructions": "Answer ONLY from the WEB context, be extremely concise, and always provide an answer—even if it wasn’t in your scraped pages."
            })
            other_future = concurrent.futures.ThreadPoolExecutor().submit(
                _call_llm, 'web', web_prompt
            )

        local_ans = local_future.result()
        other_ans = other_future.result()

        def extract(a: Any) -> Optional[str]:
            if isinstance(a, dict):
                return a.get('answer')
            if isinstance(a, str):
                return a.strip()
            return None

        la, oa = extract(local_ans), extract(other_ans)

        if la and len(la) > 50:
            final, chosen = la, 'local'
        elif la:
            final, chosen = la, 'local'
        elif oa:
            final, chosen = oa, ('world' if mode=='normal' else 'web')
        else:
            merge_prompt = json.dumps({
                "stage": "merge",
                "query": query,
                "answers": {"local": la, "other": oa},
                "instructions": "Merge both into one concise answer. If both null, return {\"answer\": null}."
            })
            merged = _call_llm('merge', merge_prompt)
            final, chosen = extract(merged), 'merged'

        if not final:
            final, chosen = "I don't have enough info.", 'none'

        result_sources: List[Dict[str, Optional[str]]] = [{"name": "Local database", "url": None}]
        if mode == "normal":
            result_sources.append({"name": "Internal LLM knowledge", "url": None})
        else:
            wiki_url = getattr(wiki_tool, 'url', None)
            if wiki_url:
                result_sources.append({"name": "Wikipedia", "url": wiki_url})
            for src in sources:
                result_sources.append(src)

        save_to_txt(final)

        duration = round(time.time() - start_time, 2)
        logger.info(f"Report done in {duration}s, source={chosen}")

        return {
            "topic": query,
            "summary": final,
            "mode": mode,
            "chosen_source": chosen,
            "latency_s": duration,
            "ask_verbose": True,
            "sources": result_sources
        }

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

genai.configure(api_key=GEMINI_API_KEY)


class ResearchAgent:
    def __init__(self, indexer):
        self.indexer = indexer
        self.cache = {}

    def generate_report(self, query: str, mode: str = "normal") -> dict:
        print(f"🔁 Mode: {mode}")

        if mode == "deep":
            model_name = "gemini-2.5-pro"
            prompt_style = "detailed"
        else:
            model_name = "gemini-2.0-flash"
            prompt_style = "fast"

        llm = genai.GenerativeModel(model_name)

        if query in self.cache:
            print("⚡ Using cached full response")
            return self.cache[query]

        local_context = self.indexer.search(query)
        local_text = "\n".join(local_context) if local_context else "[No local documents found]"

        tools_used = ["local_index"]
        web_parts = []

        # Local Summary
        local_prompt = f"""
        You are a precise research assistant. Summarize the following LOCAL context in **medium length**
        while keeping all key facts intact. Use simple language and avoid extra details or filler text.

        Return only in JSON:

        {{
          "topic": "{query}",
          "summary": "very concise summary based ONLY on local context",
          "sources": [],
          "tools_used": ["local_index"]
        }}

        Local Context:
        {local_text}
        """

        try:
            local_summary_raw = llm.generate_content(local_prompt).text
            match = re.search(r"```json\s*(\{.*?\})\s*```", local_summary_raw, re.DOTALL)
            local_summary_clean = match.group(1) if match else local_summary_raw
            local_summary = json.loads(local_summary_clean)
        except Exception as e:
            local_summary = {
                "topic": query,
                "summary": "[Error generating local summary]",
                "sources": [],
                "tools_used": ["local_index"],
                "error": str(e)
            }

        # Model (default) Summary
        default_prompt = f"""
        You are a {prompt_style} research assistant. Respond in the following JSON format only:

        {{
          "topic": "the topic string",
          "summary": "a concise and insightful explanation of the topic",
          "sources": [],
          "tools_used": []
        }}

        Topic: {query}
        """

        try:
            response = llm.generate_content(default_prompt)
            raw = response.text if hasattr(response, "text") else ""
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
            default_clean = match.group(1) if match else raw
            model_summary = json.loads(default_clean)
        except Exception as e:
            model_summary = {
                "topic": query,
                "summary": "Failed to generate a default model response.",
                "sources": [],
                "tools_used": [],
                "error": str(e)
            }

        # Final Merge using Decision Prompt
        decision_prompt = f"""
        You are an evaluator comparing two summaries for the topic "{query}".

        Local Summary:
        {local_summary['summary']}

        Model Summary:
        {model_summary['summary']}

        Rules:
        1. If the local summary contains user-provided facts or direct context, always prefer the LOCAL summary.
        2. Only merge model if it adds missing info without overriding local facts.
        3. If the local summary alone answers the query, use it and discard the model summary.
        4. Keep the answer concise and return in JSON format only.

        Return JSON ONLY:

        {{
          "topic": "{query}",
          "summary": "final combined or selected answer",
          "sources": [],
          "tools_used": []
        }}
        """

        try:
            decision_response = llm.generate_content(decision_prompt)
            raw = decision_response.text if hasattr(decision_response, "text") else ""
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
            clean = match.group(1) if match else raw
            final_output = json.loads(clean)
            final_output["tools_used"] = tools_used
            self.cache[query] = final_output
            return final_output

        except Exception as e:
            return {
                "topic": query,
                "summary": "Failed to generate final merged summary.",
                "sources": [],
                "tools_used": tools_used,
                "error": str(e)
            }

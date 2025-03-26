from langchain_core.prompts import ChatPromptTemplate

from openai_client import SUPPORT_LANGUAGES_CAST_DICT
from utils import logger


def summary_prompt(language=None):
    target_language = SUPPORT_LANGUAGES_CAST_DICT.get(language, None)
    if not target_language:
        logger.warning(f"Language '{language}' not supported for summary prompts. Defaulting to English.")
        target_language = "English"

    system_text = f"""You are a helpful AI assistant that creates structured, concise summaries.

    For the following response:
    
    If tools were used during processing:
    - Start with "I've made specific operations:" followed by bullet points (using "-") listing key actions taken (maximum 5 bullet points, one sentence each)
    - End with "Based on that, [brief conclusion]" (1-2 sentences max)
    
    If no tools were used:
    - Simply provide a short, direct answer without any special formatting
    
    If you cannot answer the request:
    - Just respond with "I'm sorry I cannot do that"
    
    Translate your entire response into {target_language} while maintaining the structure."""

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "Response to summarize: {response}")])

    return summarize_prompt
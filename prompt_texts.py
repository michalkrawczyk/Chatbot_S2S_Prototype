from langchain_core.prompts import ChatPromptTemplate

from openai_client import SUPPORT_LANGUAGES_CAST_DICT
from utils import logger


def summary_prompt(language=None):
    target_language = SUPPORT_LANGUAGES_CAST_DICT.get(language, None)
    if not target_language:
        logger.warning(f"Language '{language}' not supported for summary prompts. Defaulting to English.")
        target_language = "English"

    system_text = f"""You are a helpful AI assistant that creates structured, concise summaries.
    
    For the following response, create a summary with two clear sections:
    
    1. ACTIONS: List the key actions or analysis steps taken (maximum 10 bullet points using "-", one sentence each)
    2. CONCLUSION: Provide a very brief conclusion (1-2 sentences max) starting with "Based on that..."
    
    Translate the entire summary into {target_language} while maintaining the structure.
    
    Format your response exactly like this:
    ACTIONS:
    - [Action 1]
    - [Action 2]
    ...
    
    CONCLUSION:
    Based on that, [brief conclusion]."""

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "Response to summarize: {response}")])

    return summarize_prompt
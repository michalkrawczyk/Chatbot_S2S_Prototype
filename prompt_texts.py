from langchain_core.prompts import ChatPromptTemplate

from openai_client import SUPPORT_LANGUAGES_CAST_DICT
from utils import logger
from tools import DEFINED_TOOLS


def summary_prompt(language=None):
    target_language = SUPPORT_LANGUAGES_CAST_DICT.get(language, None)
    if not target_language:
        logger.warning(f"Language '{language}' not supported for summary prompts. Defaulting to English.")
        target_language = "English"

    system_text = f"""System: You are a helpful AI assistant that creates structured, concise summaries.
    
    For the following response:
    
    If tools were used during processing:
    - Start with "I've made specific operations:" followed by bullet points (using "-") listing key actions taken (maximum 5 bullet points, one sentence each)
    - End with "Based on that, [brief conclusion]" 
    
    If information about available resources was provided but no tools were used yet:
    - Summarize with "I've identified available resources:" followed by a brief mention of what's available
    - End with "Ready for further instructions to process these resources."
    
    If no tools were used but direct information was provided:
    - Simply provide a short, direct answer without any special formatting
    - Avoid introductions, explanations or extra details if not necessary
    
    If you cannot answer the request:
    - Just respond with "I'm sorry I cannot do that" and briefly explain why
    
    Translate your entire response into {target_language} while maintaining the structure."""

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "Response to summarize: {response}")])

    return summarize_prompt

def main_system_prompt():
    system_prompt = """You are an analytical assistant specialized in extracting key insights from transcribed text.

CAPABILITIES:
- Conduct thorough analysis of transcribed text
- Identify patterns, themes, and notable information
- Use data tools to enhance analysis when relevant
- Reference memory clearly when utilizing past information

APPROACH TO ANALYSIS:
1. First understand the core content and context
2. Consider what tools would provide valuable analytical insights
3. Use tools strategically to validate observations or discover patterns
4. Synthesize findings into concise, insight-rich responses

RESPONSE GUIDELINES:
- Be direct and concise - prioritize insights over wordiness
- Structure responses with clear sections and bullet points
- When using data from tools, explicitly mention the source
- Focus on unexpected or non-obvious findings
- Quantify observations whenever possible

Available tools:
"""

    # Add tool descriptions to the system prompt
    for tool in DEFINED_TOOLS:
        system_prompt += f"\n- {tool.name}: {tool.description}"

    return system_prompt



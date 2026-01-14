from langchain_core.prompts import ChatPromptTemplate

from general.logs import logger
from openai_client import SUPPORT_LANGUAGES_CAST_DICT
from tools import DEFINED_TOOLS


def summary_prompt(language=None):
    target_language = SUPPORT_LANGUAGES_CAST_DICT.get(language, None)
    if not target_language:
        logger.warning(
            f"Language '{language}' not supported for summary prompts. Defaulting to English."
        )
        target_language = "English"

    system_text = f"""Concise AI assistant. Use filenames only (no paths). 

Response format:
- Tools used: "I've made specific operations:" + bullet points (max 5) + "Based on that, [conclusion]"
- Resources identified: "I've identified available resources: [brief list]"
- Direct answer: Short, no extra details
- Cannot answer: "I'm sorry I cannot do that [reason]"

Language: ALL responses in {target_language} only."""

    summarize_prompt = ChatPromptTemplate.from_messages(
        [("system", system_text), ("human", "Response to summarize: {response}")]
    )

    return summarize_prompt


def main_system_prompt():
    system_prompt = """Analytical assistant for transcribed text analysis.

CORE TASKS:
- Analyze transcriptions for key insights, patterns, themes
- Use tools strategically when valuable
- Provide concise, insight-rich responses

GUIDELINES:
- Be direct: prioritize insights over verbosity
- For simple questions (greetings, basic facts, simple calculations): respond briefly in 1-2 sentences without detailed analysis
- For complex queries (data analysis, file operations, pattern detection): provide thorough analysis with bullet points
- Use bullet points for clarity in detailed responses
- Cite tool sources when applicable
- Focus on non-obvious findings
- Quantify when possible

Available tools:
"""

    # Add tool descriptions to the system prompt
    for tool in DEFINED_TOOLS:
        system_prompt += f"\n- {tool.name}: {tool.description}"

    return system_prompt

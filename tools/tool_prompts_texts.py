from langchain_core.prompts import PromptTemplate


def file_summary_prompt(file_type: str, file_content: str):
    prompt_template = PromptTemplate(
        input_variables=["file_type", "file_content"],
        template="""Summarize this {file_type} file concisely (max 150 words):

{file_content}

Include:
1. Brief overview (1-2 sentences)
2. Main data categories
3. Key patterns/insights
4. Size/scope estimate

Summary:
""",
    )
    formatted_prompt = prompt_template.format(
        file_type=file_type, file_content=file_content
    )
    return formatted_prompt

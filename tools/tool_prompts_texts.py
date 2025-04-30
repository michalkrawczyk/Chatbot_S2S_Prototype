from langchain_core.prompts import PromptTemplate


def file_summary_prompt(file_type: str, file_content: str):
    prompt_template = PromptTemplate(
        input_variables=["file_type", "file_content"],
        template="""
    You are a data analysis expert tasked with creating concise summaries of file contents.

    I have a {file_type} file with the following content:

    {file_content}

    Please provide a clear and concise summary that includes:
    1. A brief overview of what this file contains (2-3 sentences)
    2. The main categories or types of data present in the file
    3. Key patterns, trends, or insights that can be observed from a quick analysis
    4. An estimate of the size/scope of the data (number of rows/entries if applicable)

    Your summary should be professional, factual, and highlight the most important aspects of the file content.
    Limit your response to 200 words maximum.

    Summary:
    """,
    )
    formatted_prompt = prompt_template.format(
        file_type=file_type, file_content=file_content
    )
    return formatted_prompt

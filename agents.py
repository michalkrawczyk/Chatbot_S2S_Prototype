import os

import langgraph.graph as lg
import langgraph.prebuilt as lgp
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Optional, Union, Callable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from utils import logger, conditional_debug_info, RECURSION_LIMIT, AGENT_TRACE, AGENT_VERBOSE
from prompt_texts import summary_prompt


# Define agent components
class AgentState(TypedDict):
    """State object for the agent."""
    messages: List[BaseMessage]
    memory: List[Dict[str, Any]]


def should_continue(state: AgentState) -> Literal["tools", "answer_summary"]:
    messages = state['messages']
    last_message = messages[-1] # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools" # Otherwise, we route to the "answer_summary" node
    return "answer_summary"


def create_agent(model_name="o3-mini", target_language="eng"):
    """Create an agent with the specified model."""
    llm = ChatOpenAI(model=model_name, temperature=0.0) if model_name != "o3-mini" else ChatOpenAI(model=model_name)
    # Define the system prompt
    system_prompt = """You are a helpful AI assistant that analyzes transcribed text.
    When presented with transcribed text, analyze it thoroughly and provide insights.
    Use available tools when appropriate to enhance your analysis.
    When referencing information from memory, clearly indicate this in your response.
    """

    summary_llm = ChatOpenAI(model=model_name, temperature=0.0) if model_name != "o3-mini" else ChatOpenAI(model=model_name)
    summary_llm_prompt = summary_prompt(target_language)


    # Function to create messages for the agent
    def create_messages(state):
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(state["messages"])
        return messages

    # Function to call the language model
    def call_model(state):
        messages = create_messages(state)
        config = {"recursion_limit": RECURSION_LIMIT,
                    "agent_trace": AGENT_TRACE,
                    "verbose": AGENT_VERBOSE}

        conditional_debug_info(f"call_model: Calling model with messages: {messages}")
        response = llm.invoke(messages, config=config)
        return {"messages": state["messages"] + [response]}

    # Tool calling logic would go here in a more complex setup
    def call_tools(state):
        """Function to handle tool calling - placeholder for now"""
        # This is a simplified version - would implement actual tool calling logic
        return state

    def generate_summary(state):
        # Get the last AI message
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                # Extract the content from the AI message
                response_content = message.content

                # Generate a structured summary
                summary_message = summary_llm.invoke(
                    summary_llm_prompt.format(response=response_content)
                )

                # Update the state with the summary
                return {"summary": summary_message.content}

        # If no AI message found
        return {"summary": f"No response to summarize. (in {target_language})"}


    # Build the graph
    workflow = lg.StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_node("answer_summary", generate_summary)

    # Define edges
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        # First, we define the start node. We use 'agent'. # This means these are the edges taken after the 'agent' node is called.
        "agent",  # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    workflow.add_edge("tools", 'agent')
    workflow.add_edge("answer_summary", END)

    logger.info(f"Agent created with model: {model_name}")

    # Compile the graph
    return workflow.compile()


# Create the agent - will be initialized when needed
# agent_executor = None

class AgentLLM:
    _agent_executor = None
    _model_name = ""

    def initialize_agent(self, api_key, model_name="o3-mini", target_language="eng"):
        """Initialize the agent with the provided API key."""

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            self._agent_executor = create_agent(model_name=model_name, target_language=target_language)
            self._model_name = model_name
            return True
        return False


    def run_agent_on_text(self, text, memory, return_thinking=False):
        """Run the agent on the provided text."""
        if not self._agent_executor:
            return "Agent not initialized. Please provide a valid OpenAI API key."

        if not text:
            return "No text provided for analysis."

        initial_state = AgentState(
            messages=[HumanMessage(content=f"Analyze this transcribed text: {text}")],
            memory=memory or []
        )
        thinking_process_message = ""
        config = {"recursion_limit": RECURSION_LIMIT,
                    "agent_trace": AGENT_TRACE,
                    "verbose": AGENT_VERBOSE}

        try:
            result = self._agent_executor.invoke(initial_state,config=config)
            # Extract the last AI message
            if return_thinking:
               # TODO
                thinking_process_message = "Not implemented yet"


            for message in reversed(result["messages"]):
                # conditional_debug_info(f"Message: {message}")

                if isinstance(message, AIMessage):
                    return message.content, thinking_process_message

            return "No response generated."
        except Exception as e:
            return f"Error running agent: {str(e)}", thinking_process_message

    @property
    def get_agent_executor(self):
        return self._agent_executor

    @property
    def get_model_name(self):
        return self._model_name
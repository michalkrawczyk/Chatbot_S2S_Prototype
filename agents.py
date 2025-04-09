import os

import langgraph.graph as lg
import langgraph.prebuilt as lgp
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Literal, Optional, Union, Callable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from utils import logger, conditional_debug_info, RECURSION_LIMIT, AGENT_TRACE, AGENT_VERBOSE
from prompt_texts import summary_prompt, main_system_prompt
from tools import DEFINED_TOOLS_DICT


# Define agent components
class AgentState(TypedDict):
    """State object for the agent."""
    messages: List[BaseMessage]
    memory: List[Dict[str, Any]]


def should_continue_tools(state: AgentState, end_state: str = "answer_summary") -> Literal["tools", "answer_summary"]:
    messages = state['messages']
    last_message = messages[-1] # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools" # Otherwise, we route to the "answer_summary" node
    return end_state


def create_main_agent(llm, target_language="eng", summary_llm = None):
    """Create an agent with the specified model."""

    # Define the system prompt
    system_prompt = main_system_prompt()
    summary_llm = summary_llm or llm

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

    def call_tools(state):
        """Function to handle tool calling"""
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            return state

        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

            # Look up the tool in our dictionary
            if tool_name in DEFINED_TOOLS_DICT:
                tool = DEFINED_TOOLS_DICT[tool_name]
                try:
                    # Parse tool arguments if provided
                    if isinstance(tool_args, str):
                        import json
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            pass  # Use as-is if not valid JSON

                    # Call the tool with the provided arguments
                    result = tool.func(tool_args)

                    # Create a tool response message
                    tool_response = AIMessage(
                        content=str(result),
                        tool_call_id=tool_id
                    )
                    results.append(tool_response)
                except Exception as e:
                    error_message = f"Error executing tool {tool_name}: {str(e)}"
                    tool_response = AIMessage(
                        content=error_message,
                        tool_call_id=tool_id
                    )
                    results.append(tool_response)
            else:
                # Tool not found
                tool_response = AIMessage(
                    content=f"Tool '{tool_name}' not found",
                    tool_call_id=tool_id
                )
                results.append(tool_response)

        # Update state with tool responses
        return {"messages": state["messages"] + results}

    def generate_summary(state):
        # Get the last AI message
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                # Extract the content from the AI message
                response_content = message.content
                # conditional_debug_info(f"\n generate_summary: Response content: {response_content}\n")
                conditional_debug_info(f"Summary prompt: {summary_llm_prompt.format(response=response_content)}\n")

                # Generate a structured summary
                summary_message = summary_llm.invoke(
                    summary_llm_prompt.format(response=response_content)
                )
                conditional_debug_info(f"generate_summary: Summary message: {summary_message}\n")

                # Update the state with the summary
                return {"messages": state["messages"] + [summary_message]}

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
        should_continue_tools,
    )

    workflow.add_edge("tools", 'agent')
    workflow.add_edge("answer_summary", END)

    # Compile the graph
    return workflow.compile()


class AgentLLM:
    _agent_executor = None
    _model_name = ""
    _llm = None

    def initialize_agent(self, api_key, model_name="o3-mini", target_language="eng"):
        """Initialize the agent with the provided API key."""
        if model_name not in ["o3-mini", "gpt-4-turbo", "gpt-4o"]:
            logger.warning(f"Unsupported model name: {model_name}.")
            # TODO: Add later support for other models
            return False

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            self._llm = ChatOpenAI(model=model_name, temperature=0.0) if model_name != "o3-mini" else ChatOpenAI(
                model=model_name)
            self._agent_executor = create_main_agent(llm=self._llm, target_language=target_language)
            self._model_name = model_name
            logger.info(f"Main Agent created with model: {model_name}, language: {target_language}")
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

    @property
    def get_llm(self):
        return self._llm
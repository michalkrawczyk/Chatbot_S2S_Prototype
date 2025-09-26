import os

import langgraph.graph as lg
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


from general.config import RECURSION_LIMIT, AGENT_TRACE, AGENT_VERBOSE
from general.logs import logger, conditional_logger_info
from prompt_texts import summary_prompt, main_system_prompt
from tools import DEFINED_TOOLS_DICT, DEFINED_TOOLS
from openai_client import SUPPORT_LANGUAGES


# Define agent components
class AgentState(TypedDict):
    """State object for the agent."""

    messages: List[BaseMessage]
    memory: List[Dict[str, Any]]
    context: Optional[str]


def should_continue_tools(
    state: AgentState, end_state: str = "answer_summary"
) -> Literal["tools", "answer_summary"]:
    messages = state["messages"]
    last_message = messages[
        -1
    ]  # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"  # Otherwise, we route to the "answer_summary" node
    return end_state


def create_main_agent(llm, target_language="eng", summary_llm=None):
    """Create an agent with the specified model."""

    # Define the system prompt
    system_prompt = main_system_prompt()
    summary_llm = summary_llm or llm

    summary_llm_prompt = summary_prompt(target_language)

    # Function to create messages for the agent
    def create_messages(state):
        messages = [SystemMessage(content=system_prompt)]
        if state.get("context"):
            context_message = HumanMessage(
                content=f"Here is a file that might be relevant to the query:\n\n{state['context']}\n\n"
                f"Please use this information if relevant to answer the query."
            )
            messages.append(context_message)

        messages.extend(state["messages"])

        return messages

    # Function to call the language model
    def call_model(state):
        messages = create_messages(state)
        config = {
            "recursion_limit": RECURSION_LIMIT,
            "agent_trace": AGENT_TRACE,
            "verbose": AGENT_VERBOSE,
        }

        conditional_logger_info(f"call_model: Calling model with messages: {messages}")
        response = llm.invoke(messages, config=config)
        return {"messages": state["messages"] + [response]}

    def call_tools(state):
        """Function to handle tool calling"""
        messages = state["messages"]
        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return state

        results = []

        for tool_call in last_message.tool_calls:
            # Extract tool information - handle both object and dict forms
            try:
                if hasattr(tool_call, "function"):
                    # OpenAI format
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    tool_id = tool_call.id
                elif hasattr(tool_call, "name"):
                    # Direct attribute access
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    tool_id = tool_call.id
                elif isinstance(tool_call, dict):
                    # Dictionary access
                    if "function" in tool_call:
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"]["arguments"]
                        tool_id = tool_call["id"]
                    else:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args")
                        tool_id = tool_call.get("id")
                else:
                    logger.error(f"Unknown tool_call format: {tool_call}")
                    continue
            except Exception as e:
                logger.error(f"Error extracting tool information: {e}")
                continue

            # Process tool arguments
            if isinstance(tool_args, str):
                import json

                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    pass

            # Execute the tool
            result_content = ""
            if tool_name in DEFINED_TOOLS_DICT:
                tool = DEFINED_TOOLS_DICT[tool_name]
                try:
                    # Call the tool with the appropriate arguments
                    if hasattr(tool, "func"):
                        # If the tool has a func attribute (like a structured tool)
                        if isinstance(tool_args, dict):
                            result = tool.func(**tool_args)
                        else:
                            result = tool.func(tool_args)
                    elif callable(tool):
                        # If the tool itself is callable
                        if isinstance(tool_args, dict):
                            result = tool(**tool_args)
                        else:
                            result = tool(tool_args)
                    else:
                        raise ValueError(f"Tool {tool_name} is not callable")

                    result_content = str(result)
                except Exception as e:
                    result_content = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(result_content)
            else:
                result_content = f"Tool '{tool_name}' not found"
                logger.warning(result_content)

            # Create a proper tool response message
            try:
                # Try importing ToolMessage (newer LangChain versions)
                from langchain_core.messages import ToolMessage

                tool_response = ToolMessage(
                    content=result_content, tool_call_id=tool_id
                )
            except (ImportError, AttributeError):
                # Fall back to AIMessage format
                tool_response = AIMessage(
                    content=result_content, additional_kwargs={"tool_call_id": tool_id}
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
                conditional_logger_info(
                    f"Summary prompt: {summary_llm_prompt.format(response=response_content)}\n"
                )

                # Generate a structured summary
                summary_message = summary_llm.invoke(
                    summary_llm_prompt.format(response=response_content)
                )
                conditional_logger_info(
                    f"generate_summary: Summary message: {summary_message}\n"
                )

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

    workflow.add_edge("tools", "agent")
    workflow.add_edge("answer_summary", END)

    # Compile the graph
    return workflow.compile()


class AgentLLM:
    _agent_executor = None
    _model_name = ""
    _llm = None
    _llm_with_tools = None
    _context = ""
    _summary_language = "eng"

    def initialize_agent(self, api_key, model_name="gpt-4.1-mini"):
        """Initialize the agent with the provided API key."""
        if model_name not in ["o3-mini", "gpt-4-turbo", "gpt-4o", "gpt-4.1-mini", "gpt-5-mini"]:
            logger.warning(f"Unsupported model name: {model_name}.")
            # TODO: Add later support for other models
            return False

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            self._llm = (
                ChatOpenAI(model=model_name, temperature=0.0)
                if model_name != "o3-mini"
                else ChatOpenAI(model=model_name)
            )

            self._llm_with_tools = self._llm.bind_tools(DEFINED_TOOLS)
            # TODO: Check if work correctly with other chats after implementing them
            # TODO: node with tools should be inside main agent?
            self._agent_executor = create_main_agent(
                llm=self._llm_with_tools,
                target_language=self._summary_language,
                summary_llm=self._llm,
            )
            self._model_name = model_name
            logger.info(
                f"Main Agent created with model: {model_name}, language: {self._summary_language}"
            )
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
            memory=memory or [],
            context=(
                self._prepare_context_message(self._context[0], self._context[1])
                if self._context
                else None
            ),
        )
        thinking_process_message = ""
        config = {
            "recursion_limit": RECURSION_LIMIT,
            "agent_trace": AGENT_TRACE,
            "verbose": AGENT_VERBOSE,
        }

        try:
            result = self._agent_executor.invoke(initial_state, config=config)
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

    def _prepare_context_message(self, context: str, context_type: str = "file info"):
        """Prepare the context message for the agent."""
        context_messages = {
            "file info": f"Here is a file that might be relevant to the query:\n\n{context}\n\n"
            f"Please check if this file contains information relevant to the query before exploring other sources.",
            "file content": f"Here is the content of the file:\n\n{context}\n\n"
            f"Please check if this information is relevant to the query before exploring other sources.",
            "general": f"Here is some additional information that might be relevant to the query:\n\n{context}\n\n"
            f"Please use this information if relevant to answer the query.",
        }
        if context_type not in context_messages:
            logger.warning(
                f"Unknown context type: {context_type}. Defaulting to 'general'."
            )

        # Return the appropriate message or the raw context if context_type is not in the dictionary
        return context_messages.get(context_type, context)

    def set_context(self, context: str, context_type: str = "file info"):
        """Set the context for the agent."""
        # TODO: Add awerness of length of context (maximum token size)

        logger.info(f"Setting context for the agent: {context_type} - {context}")
        self._context = (context, context_type)

    def change_summary_language(self, language: str):
        """Set the language for the summary."""
        if language in SUPPORT_LANGUAGES:
            self._summary_language = language
            logger.info(f"Summary language set to: {language}")
        else:
            logger.warning(f"Unsupported summary language: {language}.")
            return

        if self._agent_executor is not None:
            self._agent_executor = create_main_agent(
                llm=self._llm_with_tools,
                target_language=self._summary_language,
                summary_llm=self._llm,
            )
            logger.info(f"Agent executor updated with new summary language: {language}")


from langchain_core.runnables import Runnable, RunnableConfig
from typing import Dict
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage



# Defining the State of the graph which contains an append-only list of messages. (chat history)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Defining the Assistant class which takes the Graph state, formats it into a prompt and then invokes the LLM.
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # Configuration for user_id (which is same as thread_id)
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_info": user_id}
            result = self.runnable.invoke(state)

            # If the LLM returns an empty response, we will re-prompt it again.
            if not getattr(result, 'tool_calls', None) and (
                not getattr(result, 'content', None)
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}



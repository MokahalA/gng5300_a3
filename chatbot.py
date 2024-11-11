from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import tools_condition
import uuid
from assistant import State, Assistant
from tools import (
    get_product_categories,
    search_product_by_name,
    get_recommendations,
    add_to_cart,
    remove_from_cart,
    view_cart,
    get_delivery_time,
    get_returns_policy,
    get_shipping_policy,
    get_payment_methods,
)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

# Helper function to print the assistant's response
def _print_event(event: dict, _printed: set, max_length=1500):
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        
        # Only print if the message is an AIMessage (default behavior)
        if isinstance(message, AIMessage) and message.id not in _printed:
            # Extract just the content from the AI message
            content = message.content
            if len(content) > max_length:
                content = content[:max_length] + " ... (truncated)"
            print(content)
            print("\n")  # Add divider after the response
            _printed.add(message.id)
            

# Handler for tool error messages.
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Create a tool node with a fallback to handle errors
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Main function to run the chatbot
def main():
    # Generate a unique thread ID for the conversation session (same as user_id)
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": thread_id,
        }
    }

    # Initialize the LLM model
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=1,
    )

    assistant_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful customer support assistant for the Skincare Products company."
            "Reply in a friendly way whenever the user says Hello or Hi and Greet them in your first response."
            "Provide a good detailed response to the questions you are asked."
            "Do not provide answers about products that are outside the tools available to you such as the database."
            "If a tool returns an empty response, kindly ask the user to rephrase their question or provide more details."
        ),
        ("placeholder", "{messages}")
    ])

    # Define the tools that the assistant can use safely without user confirmation
    safe_tools = [
        get_product_categories,
        search_product_by_name,
        get_recommendations,
        view_cart,
        get_delivery_time,
        get_returns_policy,
        get_shipping_policy,
        get_payment_methods,
    ]

    # Define the sensitive tools that require user confirmation before execution
    sensitive_tools = [add_to_cart, remove_from_cart]

    sensitive_tool_names = {t.name for t in sensitive_tools}
    assistant_runnable = assistant_prompt | llm.bind_tools(safe_tools + sensitive_tools)

    # Route tools based on input State
    def route_tools(state: State):
        """Route tools based on AI response."""
        next_node = tools_condition(state)
        if next_node == END:
            return END

        ai_message = state["messages"][-1]
        if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
            first_tool_call = ai_message.tool_calls[0]
            print("\n[DEBUG] Tool call detected:", first_tool_call)
            if first_tool_call["name"] in sensitive_tool_names:
                return "sensitive_tools"
        return "safe_tools"

    # Build the state graph
    builder = StateGraph(State)
    builder.add_node("skincare_assistant", Assistant(assistant_runnable))
    builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
    builder.add_edge(START, "skincare_assistant")
    builder.add_node("sensitive_tools", create_tool_node_with_fallback(sensitive_tools))
    builder.add_conditional_edges(
        "skincare_assistant", route_tools, ["safe_tools", "sensitive_tools", END]
    )
    builder.add_edge("safe_tools", "skincare_assistant")
    builder.add_edge("sensitive_tools", "skincare_assistant")

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["sensitive_tools"],    
    )

    ## Uncomment to generate the graph diagram
    # try:
    #     img = graph.get_graph(xray=True).draw_mermaid_png()
    #     with open("graph_diagram.png", "wb") as f:
    #         f.write(img)
    # except Exception as e:
    #     print("An error occurred:", e)
    #     pass


    # Start the conversation loop
    _printed = set()
    print("Welcome to the Skincare Assistant Chatbot!")

    while True:
        # Get user input
        print("-" * 50)
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nThank you for using Beauty Products Customer Support. Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue

        # Process the user input
        print("\nAssistant:", end=" ")
        events = graph.stream(
            {"messages": ("user", user_input)}, 
            config, 
            stream_mode="values"
        )
        
        # Print each event response from the assistant
        for event in events:
            _print_event(event, _printed)


        # Get the graph state after the user input
        snapshot = graph.get_state(config)

        # Handle any interrupts (like adding/removing items) that need confirmation
        while snapshot.next:
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
            except:
                user_input = "y"
            if user_input.strip() == "y":
                # Just continue
                result = graph.invoke(
                    None,
                    config,
                )
                
                print(result['messages'][-1].content)
                
            else:
                # Satisfy the tool invocation by providing a user message
                result = graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = graph.get_state(config)

if __name__ == "__main__":
    main()
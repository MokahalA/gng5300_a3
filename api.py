"""
FastAPI Backend for the Skincare Chatbot
Exposes the LangGraph chatbot as a REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import sqlite3
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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
from vector_search import semantic_product_search, get_vector_search


# Preload models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload the vector search model
    print("🔄 Preloading vector search model...")
    get_vector_search()
    print("✅ Vector search model loaded!")
    yield
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(title="Skincare Chatbot API", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for active sessions
sessions: Dict[str, Any] = {}

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: List[ToolCall] = []
    requires_confirmation: bool = False
    pending_action: Optional[Dict[str, Any]] = None

class ConfirmRequest(BaseModel):
    session_id: str
    approved: bool
    user_message: Optional[str] = None

class CartItem(BaseModel):
    product_id: int
    product_name: str
    price: float
    quantity: int

class CartResponse(BaseModel):
    items: List[CartItem]
    total_price: float


def create_graph():
    """Create and return a new LangGraph instance."""
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0.1,
    )

    assistant_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a customer support assistant for a Skincare Products store.

TOOL SELECTION - VERY IMPORTANT:
- For skin concerns (oily, dry, acne, sensitive, aging): USE semantic_product_search
- For specific product names: USE search_product_by_name
- For cart actions: USE add_to_cart, remove_from_cart, view_cart

EXAMPLES:
- "I have oily skin" → semantic_product_search(query="oily skin")
- "products for dry skin under $30" → semantic_product_search(query="dry skin", max_price=30)
- "find the Vitamin C Serum" → search_product_by_name(product_name="Vitamin C")

RULES:
- Always use the products returned by tools - never make up products
- If tool returns empty or error, tell the user honestly
- Be concise in responses

CATEGORIES: Moisturizers, Cleansers, Serums, Sunscreens, Masks, Toners"""
        ),
        ("placeholder", "{messages}")
    ])

    safe_tools = [
        get_product_categories,
        search_product_by_name,
        get_recommendations,
        semantic_product_search,
        view_cart,
        get_delivery_time,
        get_returns_policy,
        get_shipping_policy,
        get_payment_methods,
    ]

    sensitive_tools = [add_to_cart, remove_from_cart]
    sensitive_tool_names = {t.name for t in sensitive_tools}
    
    assistant_runnable = assistant_prompt | llm.bind_tools(safe_tools + sensitive_tools)

    def handle_tool_error(state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def create_tool_node_with_fallback(tools: list) -> dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(handle_tool_error)], exception_key="error"
        )

    def route_tools(state: State):
        next_node = tools_condition(state)
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
            first_tool_call = ai_message.tool_calls[0]
            if first_tool_call["name"] in sensitive_tool_names:
                return "sensitive_tools"
        return "safe_tools"

    builder = StateGraph(State)
    builder.add_node("skincare_assistant", Assistant(assistant_runnable))
    builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
    builder.add_node("sensitive_tools", create_tool_node_with_fallback(sensitive_tools))
    builder.add_edge(START, "skincare_assistant")
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
    
    return graph


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Any]:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {
        "graph": create_graph(),
        "config": {
            "configurable": {
                "thread_id": new_id,
                "user_id": new_id,
            }
        }
    }
    return new_id, sessions[new_id]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the chatbot."""
    session_id, session = get_or_create_session(request.session_id)
    graph = session["graph"]
    config = session["config"]
    
    # Check for greetings
    greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                     'good evening', 'greetings', 'howdy', 'hiya']
    is_greeting = any(
        request.message.lower().strip() == greet or 
        request.message.lower().strip().startswith(greet + ' ') or
        request.message.lower().strip().startswith(greet + ',') or
        request.message.lower().strip().startswith(greet + '!')
        for greet in greeting_words
    )
    
    if is_greeting:
        return ChatResponse(
            response="Hello! Welcome to our Skincare Products store. How can I help you today? I can help you find products for your skin type, check prices, add items to your cart, or answer questions about shipping and returns.",
            session_id=session_id,
            tool_calls=[],
            requires_confirmation=False
        )
    
    # Process through the graph
    tool_calls_made = []
    response_text = ""
    
    events = graph.stream(
        {"messages": ("user", request.message)},
        config,
        stream_mode="values"
    )
    
    for event in events:
        messages = event.get("messages", [])
        if messages:
            last_msg = messages[-1] if isinstance(messages, list) else messages
            # Debug: print all messages
            print(f"[DEBUG] Message type: {type(last_msg).__name__}")
            if isinstance(last_msg, AIMessage):
                response_text = last_msg.content
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"[DEBUG] Tool call: {tc['name']} with args {tc['args']}")
                        tool_calls_made.append(ToolCall(
                            name=tc["name"],
                            args=tc["args"],
                            id=tc["id"]
                        ))
            elif isinstance(last_msg, ToolMessage):
                print(f"[DEBUG] Tool result: {last_msg.content[:200]}...")
    
    # Check if we need confirmation (interrupted before sensitive_tools)
    snapshot = graph.get_state(config)
    requires_confirmation = bool(snapshot.next)
    pending_action = None
    
    if requires_confirmation and tool_calls_made:
        pending_action = {
            "tool": tool_calls_made[-1].name,
            "args": tool_calls_made[-1].args
        }
    
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        tool_calls=tool_calls_made,
        requires_confirmation=requires_confirmation,
        pending_action=pending_action
    )


@app.post("/confirm", response_model=ChatResponse)
async def confirm_action(request: ConfirmRequest):
    """Confirm or deny a pending action."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    graph = session["graph"]
    config = session["config"]
    
    tool_calls_made = []
    response_text = ""
    
    if request.approved:
        # Continue with the action
        result = graph.invoke(None, config)
        if result and "messages" in result:
            last_msg = result["messages"][-1]
            response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    else:
        # Cancel the action
        snapshot = graph.get_state(config)
        messages = snapshot.values.get("messages", [])
        if messages:
            last_ai_msg = messages[-1]
            if hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
                result = graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=last_ai_msg.tool_calls[0]["id"],
                                content=f"Action cancelled by user. Reason: {request.user_message or 'No reason provided'}",
                            )
                        ]
                    },
                    config,
                )
                if result and "messages" in result:
                    response_text = result["messages"][-1].content
    
    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        tool_calls=tool_calls_made,
        requires_confirmation=False
    )


@app.get("/cart/{session_id}", response_model=CartResponse)
async def get_cart(session_id: str):
    """Get the current cart contents for a session."""
    db = "skincare.sqlite"
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT product_id, product_name, price, quantity FROM shopping_carts WHERE user_id = ?",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    items = [
        CartItem(
            product_id=row[0],
            product_name=row[1],
            price=row[2],
            quantity=row[3]
        )
        for row in rows
    ]
    
    total = sum(item.price * item.quantity for item in items)
    
    return CartResponse(items=items, total_price=total)


@app.delete("/cart/{session_id}")
async def clear_cart(session_id: str):
    """Clear the cart for a session."""
    db = "skincare.sqlite"
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM shopping_carts WHERE user_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"message": "Cart cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

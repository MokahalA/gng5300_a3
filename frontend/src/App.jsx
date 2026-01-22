import { useState, useEffect, useRef } from 'react'

const API_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [cart, setCart] = useState({ items: [], total_price: 0 })
  const [debugLogs, setDebugLogs] = useState([])
  const [showDebug, setShowDebug] = useState(true)
  const [pendingConfirmation, setPendingConfirmation] = useState(null)
  
  const messagesEndRef = useRef(null)
  const debugEndRef = useRef(null)

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Scroll to bottom of debug logs
  useEffect(() => {
    debugEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [debugLogs])

  // Fetch cart when session changes
  useEffect(() => {
    if (sessionId) {
      fetchCart()
    }
  }, [sessionId])

  const fetchCart = async () => {
    if (!sessionId) return
    try {
      const res = await fetch(`${API_URL}/cart/${sessionId}`)
      const data = await res.json()
      setCart(data)
    } catch (err) {
      console.error('Failed to fetch cart:', err)
    }
  }

  const addDebugLog = (toolCalls) => {
    if (toolCalls && toolCalls.length > 0) {
      const newLogs = toolCalls.map(tc => ({
        timestamp: new Date().toLocaleTimeString(),
        toolName: tc.name,
        args: tc.args
      }))
      setDebugLogs(prev => [...prev, ...newLogs])
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId
        })
      })

      const data = await res.json()
      setSessionId(data.session_id)

      // Log tool calls
      addDebugLog(data.tool_calls)

      if (data.requires_confirmation) {
        setPendingConfirmation(data.pending_action)
        setMessages(prev => [...prev, {
          role: 'system',
          content: `⚠️ Action requires confirmation: ${data.pending_action.tool} with ${JSON.stringify(data.pending_action.args)}`
        }])
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
        // Refresh cart after each response
        setTimeout(fetchCart, 500)
      }
    } catch (err) {
      console.error('Error:', err)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, something went wrong. Please try again.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleConfirm = async (approved) => {
    if (!pendingConfirmation) return
    setIsLoading(true)

    try {
      const res = await fetch(`${API_URL}/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          approved: approved,
          user_message: approved ? null : 'User cancelled the action'
        })
      })

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
      setPendingConfirmation(null)
      
      // Refresh cart
      setTimeout(fetchCart, 500)
    } catch (err) {
      console.error('Error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      {/* Chat Section */}
      <div className="chat-section">
        <div className="chat-header">
          <h1>🧴 Skincare Assistant</h1>
          <p>Your personal skincare shopping assistant</p>
        </div>

        {sessionId && (
          <div className="session-info">
            Session: <code>{sessionId.substring(0, 8)}...</code>
          </div>
        )}

        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="message assistant">
              Hello! Welcome to our Skincare Products store. How can I help you today?
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.content}
            </div>
          ))}
          
          {isLoading && (
            <div className="loading">
              <div className="loading-dot"></div>
              <div className="loading-dot"></div>
              <div className="loading-dot"></div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {pendingConfirmation && (
          <div className="confirmation-dialog">
            <p>Do you want to proceed with this action?</p>
            <div className="confirmation-buttons">
              <button className="confirm-btn" onClick={() => handleConfirm(true)}>
                ✓ Yes, proceed
              </button>
              <button className="cancel-btn" onClick={() => handleConfirm(false)}>
                ✗ Cancel
              </button>
            </div>
          </div>
        )}

        <div className="chat-input-container">
          <input
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about products, add to cart, check policies..."
            disabled={isLoading || pendingConfirmation}
          />
          <button 
            className="send-button" 
            onClick={sendMessage}
            disabled={isLoading || !input.trim() || pendingConfirmation}
          >
            Send
          </button>
        </div>
      </div>

      {/* Sidebar */}
      <div className="sidebar">
        {/* Cart Section */}
        <div className="sidebar-section cart-section">
          <h2>🛒 Shopping Cart</h2>
          {cart.items.length === 0 ? (
            <p className="cart-empty">Your cart is empty</p>
          ) : (
            <>
              <div className="cart-items">
                {cart.items.map((item, idx) => (
                  <div key={idx} className="cart-item">
                    <span className="cart-item-name">{item.product_name}</span>
                    <span className="cart-item-qty">×{item.quantity}</span>
                    <span className="cart-item-price">${(item.price * item.quantity).toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <div className="cart-total">
                <span>Total:</span>
                <span className="cart-total-amount">${cart.total_price.toFixed(2)}</span>
              </div>
            </>
          )}
        </div>

        {/* Debug Section */}
        <div className="sidebar-section debug-section">
          <h2>🔧 Debug Panel</h2>
          <div className="debug-toggle">
            <input
              type="checkbox"
              id="showDebug"
              checked={showDebug}
              onChange={(e) => setShowDebug(e.target.checked)}
            />
            <label htmlFor="showDebug">Show tool calls</label>
          </div>
          
          {showDebug && (
            <div className="debug-logs">
              {debugLogs.length === 0 ? (
                <div style={{ color: '#666' }}>No tool calls yet...</div>
              ) : (
                debugLogs.map((log, idx) => (
                  <div key={idx} className="debug-entry">
                    <div className="debug-timestamp">{log.timestamp}</div>
                    <div className="debug-tool-name">{log.toolName}</div>
                    <div className="debug-args">
                      {JSON.stringify(log.args, null, 2)}
                    </div>
                  </div>
                ))
              )}
              <div ref={debugEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

/**
 * THAU Code Desktop - Chat Interface Component
 */

import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage, AgentRole } from '@/types';
import { wsService } from '@/services/websocket';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onMessageSent: (message: ChatMessage) => void;
  selectedAgent: AgentRole;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onMessageSent,
  selectedAgent,
}) => {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    // Create user message
    const userMessage: ChatMessage = {
      id: `${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    };

    onMessageSent(userMessage);

    // Send to WebSocket
    wsService.sendMessage(input, selectedAgent);

    setInput('');
    setIsTyping(true);

    // Stop typing indicator after 30 seconds
    setTimeout(() => setIsTyping(false), 30000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Messages Container */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
          backgroundColor: '#1e1e1e',
        }}
      >
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isTyping && (
          <div style={{ padding: '10px', color: '#888' }}>
            <em>THAU is thinking...</em>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Container */}
      <div
        style={{
          padding: '20px',
          backgroundColor: '#252525',
          borderTop: '1px solid #3e3e3e',
        }}
      >
        <div style={{ display: 'flex', gap: '10px' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={`Ask ${selectedAgent}...`}
            style={{
              flex: 1,
              padding: '12px',
              backgroundColor: '#1e1e1e',
              color: '#d4d4d4',
              border: '1px solid #3e3e3e',
              borderRadius: '4px',
              fontSize: '14px',
              resize: 'none',
              fontFamily: 'monospace',
              minHeight: '60px',
            }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            style={{
              padding: '12px 24px',
              backgroundColor: input.trim() ? '#007acc' : '#3e3e3e',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: input.trim() ? 'pointer' : 'not-allowed',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            Send
          </button>
        </div>
        <div style={{ marginTop: '8px', fontSize: '12px', color: '#888' }}>
          Active Agent: <strong style={{ color: '#007acc' }}>{selectedAgent}</strong>
        </div>
      </div>
    </div>
  );
};

/**
 * Message Bubble Component
 */
const MessageBubble: React.FC<{ message: ChatMessage }> = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div
      style={{
        marginBottom: '16px',
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <div
        style={{
          maxWidth: '80%',
          padding: '12px 16px',
          backgroundColor: isUser ? '#007acc' : isSystem ? '#2d2d2d' : '#2d5a2d',
          borderRadius: '8px',
          color: '#fff',
        }}
      >
        {/* Header */}
        <div
          style={{
            fontSize: '11px',
            color: '#aaa',
            marginBottom: '6px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>
            {message.role === 'user' ? 'You' : message.agent_role || 'THAU'}
          </span>
          <span>
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>

        {/* Content */}
        <div style={{ fontSize: '14px', lineHeight: '1.5' }}>
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Thinking (if present) */}
        {message.thinking && (
          <details style={{ marginTop: '8px', fontSize: '12px', color: '#ccc' }}>
            <summary style={{ cursor: 'pointer' }}>Reasoning</summary>
            <div style={{ marginTop: '4px', fontStyle: 'italic' }}>
              {message.thinking}
            </div>
          </details>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;

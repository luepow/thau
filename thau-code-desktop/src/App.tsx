/**
 * THAU Code Desktop - Main Application Component
 */

import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import AgentPanel from './components/AgentPanel';
import PlannerView from './components/PlannerView';
import ToolFactory from './components/ToolFactory';
import CodeEditor from './components/CodeEditor';
import { ChatMessage, AgentRole } from './types';
import { wsService } from './services/websocket';
import { apiService } from './services/api';

type View = 'chat' | 'agents' | 'planner' | 'tools' | 'editor';

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<View>('chat');
  const [connected, setConnected] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<AgentRole>(AgentRole.GENERAL);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [code, setCode] = useState('# Welcome to THAU Code\n\nprint("Hello, THAU!")');

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Get backend URL from Electron
      const backendUrl = await (window as any).electronAPI?.getBackendUrl() || 'http://localhost:8001';

      // Configure services
      apiService.setBaseURL(backendUrl);
      wsService.setBaseURL(backendUrl.replace('http://', 'ws://'));
      wsService.setSessionId(`session_${Date.now()}`);

      // Set up WebSocket listeners
      wsService.onMessage((message) => {
        setMessages((prev) => [...prev, message]);
      });

      wsService.onError((error) => {
        console.error('WebSocket error:', error);
        setMessages((prev) => [
          ...prev,
          {
            id: `${Date.now()}`,
            role: 'system',
            content: `Error: ${error}`,
            timestamp: new Date().toISOString(),
          },
        ]);
      });

      wsService.onConnection((isConnected) => {
        setConnected(isConnected);
        if (isConnected) {
          setMessages((prev) => [
            ...prev,
            {
              id: `${Date.now()}`,
              role: 'system',
              content: 'Connected to THAU Code Server',
              timestamp: new Date().toISOString(),
            },
          ]);
        }
      });

      // Connect WebSocket
      wsService.connect();
    } catch (error) {
      console.error('Failed to initialize app:', error);
    }
  };

  const handleMessageSent = (message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  };

  const getViewIcon = (view: View): string => {
    const icons: Record<View, string> = {
      chat: '💬',
      agents: '🤖',
      planner: '📋',
      tools: '🔧',
      editor: '📝',
    };
    return icons[view];
  };

  const getViewTitle = (view: View): string => {
    const titles: Record<View, string> = {
      chat: 'Chat',
      agents: 'Agents',
      planner: 'Planner',
      tools: 'Tools',
      editor: 'Editor',
    };
    return titles[view];
  };

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        backgroundColor: '#1e1e1e',
        color: '#d4d4d4',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
      {/* Sidebar */}
      <div
        style={{
          width: sidebarOpen ? '60px' : '0',
          backgroundColor: '#252525',
          borderRight: '1px solid #3e3e3e',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          transition: 'width 0.3s',
        }}
      >
        {/* Logo */}
        <div
          style={{
            padding: '16px',
            textAlign: 'center',
            borderBottom: '1px solid #3e3e3e',
            fontSize: '24px',
          }}
        >
          🧠
        </div>

        {/* Navigation */}
        <div style={{ flex: 1, padding: '8px 0' }}>
          {(['chat', 'agents', 'planner', 'tools', 'editor'] as View[]).map((view) => (
            <button
              key={view}
              onClick={() => setActiveView(view)}
              title={getViewTitle(view)}
              style={{
                width: '100%',
                padding: '12px',
                backgroundColor: activeView === view ? '#007acc' : 'transparent',
                color: '#fff',
                border: 'none',
                fontSize: '20px',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => {
                if (activeView !== view) {
                  e.currentTarget.style.backgroundColor = '#3e3e3e';
                }
              }}
              onMouseLeave={(e) => {
                if (activeView !== view) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              {getViewIcon(view)}
            </button>
          ))}
        </div>

        {/* Connection Status */}
        <div
          style={{
            padding: '16px',
            textAlign: 'center',
            borderTop: '1px solid #3e3e3e',
          }}
        >
          <div
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: connected ? '#2d7a2d' : '#8b2e2e',
              margin: '0 auto',
            }}
            title={connected ? 'Connected' : 'Disconnected'}
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <div
          style={{
            padding: '12px 20px',
            backgroundColor: '#252525',
            borderBottom: '1px solid #3e3e3e',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{
                padding: '6px 12px',
                backgroundColor: '#3e3e3e',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
              }}
            >
              ☰
            </button>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold' }}>
              THAU Code
            </h1>
            <div
              style={{
                fontSize: '12px',
                padding: '4px 8px',
                backgroundColor: '#3e3e3e',
                borderRadius: '3px',
                color: '#aaa',
              }}
            >
              {getViewTitle(activeView)}
            </div>
          </div>

          <div style={{ fontSize: '12px', color: '#888' }}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </div>
        </div>

        {/* View Content */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {activeView === 'chat' && (
            <>
              <div style={{ flex: 1 }}>
                <ChatInterface
                  messages={messages}
                  onMessageSent={handleMessageSent}
                  selectedAgent={selectedAgent}
                />
              </div>
              <div style={{ width: '300px', borderLeft: '1px solid #3e3e3e' }}>
                <AgentPanel
                  selectedAgent={selectedAgent}
                  onAgentSelect={setSelectedAgent}
                />
              </div>
            </>
          )}

          {activeView === 'agents' && (
            <div style={{ flex: 1 }}>
              <AgentPanel
                selectedAgent={selectedAgent}
                onAgentSelect={setSelectedAgent}
              />
            </div>
          )}

          {activeView === 'planner' && (
            <div style={{ flex: 1 }}>
              <PlannerView />
            </div>
          )}

          {activeView === 'tools' && (
            <div style={{ flex: 1 }}>
              <ToolFactory />
            </div>
          )}

          {activeView === 'editor' && (
            <div style={{ flex: 1, padding: '20px' }}>
              <CodeEditor
                value={code}
                onChange={(value) => setCode(value || '')}
                language="python"
                height="calc(100vh - 120px)"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;

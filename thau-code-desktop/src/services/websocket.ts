/**
 * THAU Code Desktop - WebSocket Service
 *
 * Manages real-time communication with THAU Code Server
 */

import { ChatMessage, WSMessage, AgentRole } from '@/types';

type MessageCallback = (message: ChatMessage) => void;
type ErrorCallback = (error: string) => void;
type ConnectionCallback = (connected: boolean) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private baseURL: string;
  private sessionId: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000; // ms

  private messageCallbacks: MessageCallback[] = [];
  private errorCallbacks: ErrorCallback[] = [];
  private connectionCallbacks: ConnectionCallback[] = [];

  constructor(baseURL: string = 'ws://localhost:8001', sessionId: string = 'default') {
    this.baseURL = baseURL;
    this.sessionId = sessionId;
  }

  /**
   * Connect to WebSocket server
   */
  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    const wsUrl = `${this.baseURL}/ws/chat/${this.sessionId}`;
    console.log(`Connecting to WebSocket: ${wsUrl}`);

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.notifyConnectionCallbacks(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const data: WSMessage = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.notifyErrorCallbacks('WebSocket connection error');
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.notifyConnectionCallbacks(false);
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.notifyErrorCallbacks('Failed to connect to server');
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Send chat message
   */
  sendMessage(content: string, agentRole: AgentRole = AgentRole.GENERAL) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.notifyErrorCallbacks('Not connected to server');
      return;
    }

    const message: WSMessage = {
      type: 'message',
      content,
      agent_role: agentRole,
    };

    this.ws.send(JSON.stringify(message));
  }

  /**
   * Invoke MCP tool
   */
  invokeTool(toolName: string, args: Record<string, any>) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.notifyErrorCallbacks('Not connected to server');
      return;
    }

    const message: WSMessage = {
      type: 'tool_call',
      tool_name: toolName,
      arguments: args,
    };

    this.ws.send(JSON.stringify(message));
  }

  /**
   * Handle incoming message
   */
  private handleMessage(data: WSMessage) {
    if (data.type === 'message') {
      const chatMessage: ChatMessage = {
        id: data.task_id || `${Date.now()}`,
        role: 'agent',
        content: data.content || '',
        agent_role: data.agent_role as AgentRole,
        timestamp: data.timestamp || new Date().toISOString(),
        thinking: data.thinking,
      };

      this.notifyMessageCallbacks(chatMessage);
    } else if (data.type === 'tool_result') {
      // Handle tool result
      const resultMessage: ChatMessage = {
        id: `${Date.now()}`,
        role: 'system',
        content: data.success
          ? `Tool '${data.tool_name}' completed successfully`
          : `Tool '${data.tool_name}' failed: ${data.error}`,
        timestamp: data.timestamp || new Date().toISOString(),
        metadata: {
          toolName: data.tool_name,
          result: data.result,
        },
      };

      this.notifyMessageCallbacks(resultMessage);
    } else if (data.type === 'error') {
      this.notifyErrorCallbacks(data.error || 'Unknown error');
    }
  }

  /**
   * Attempt to reconnect
   */
  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached');
      this.notifyErrorCallbacks('Failed to reconnect to server');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * this.reconnectAttempts;

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Callback registration
   */
  onMessage(callback: MessageCallback) {
    this.messageCallbacks.push(callback);
  }

  onError(callback: ErrorCallback) {
    this.errorCallbacks.push(callback);
  }

  onConnection(callback: ConnectionCallback) {
    this.connectionCallbacks.push(callback);
  }

  /**
   * Notify callbacks
   */
  private notifyMessageCallbacks(message: ChatMessage) {
    this.messageCallbacks.forEach(cb => cb(message));
  }

  private notifyErrorCallbacks(error: string) {
    this.errorCallbacks.forEach(cb => cb(error));
  }

  private notifyConnectionCallbacks(connected: boolean) {
    this.connectionCallbacks.forEach(cb => cb(connected));
  }

  /**
   * Get connection status
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Update configuration
   */
  setBaseURL(url: string) {
    this.baseURL = url.replace('http://', 'ws://').replace('https://', 'wss://');
  }

  setSessionId(sessionId: string) {
    this.sessionId = sessionId;
  }
}

// Export singleton instance
export const wsService = new WebSocketService();

export default WebSocketService;

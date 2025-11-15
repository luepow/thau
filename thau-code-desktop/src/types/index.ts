/**
 * THAU Code Desktop - Type Definitions
 */

// Agent Types
export enum AgentRole {
  GENERAL = 'general',
  CODE_WRITER = 'code_writer',
  CODE_REVIEWER = 'code_reviewer',
  DEBUGGER = 'debugger',
  RESEARCHER = 'researcher',
  PLANNER = 'planner',
  ARCHITECT = 'architect',
  TESTER = 'tester',
  DOCUMENTER = 'documenter',
  API_SPECIALIST = 'api_specialist',
  DATA_ANALYST = 'data_analyst',
  SECURITY = 'security',
  VISUAL_CREATOR = 'visual_creator',
}

export interface Agent {
  id: string;
  role: AgentRole;
  name: string;
  description: string;
  status: 'active' | 'idle';
}

// Task Types
export interface Task {
  id: string;
  description: string;
  agent_role: AgentRole;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: string;
  created_at?: string;
  completed_at?: string;
}

// Plan Types
export enum TaskComplexity {
  TRIVIAL = 'trivial',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  VERY_HIGH = 'very_high',
}

export enum TaskPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface PlanStep {
  step_number: number;
  description: string;
  action_type: string;
  estimated_effort: string;
  dependencies: number[];
  tools_needed: string[];
}

export interface Plan {
  task_description: string;
  complexity: TaskComplexity;
  priority: TaskPriority;
  estimated_hours: number;
  steps: PlanStep[];
  risks: string[];
  assumptions: string[];
  success_criteria: string[];
}

// MCP Tool Types
export interface MCPParameter {
  name: string;
  type: string;
  description: string;
  required?: boolean;
  enum?: any[];
}

export interface MCPTool {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: {
      type: string;
      properties: Record<string, any>;
      required: string[];
    };
  };
}

export interface MCPToolResult {
  call_id: string;
  success: boolean;
  result: any;
  error?: string;
  execution_time_ms?: number;
}

// Tool Factory Types
export interface GeneratedTool {
  name: string;
  description: string;
  category: string;
  parameters: MCPParameter[];
  file_path?: string;
  created_at?: string;
}

// Chat Message Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'agent' | 'system';
  content: string;
  agent_role?: AgentRole;
  timestamp: string;
  thinking?: string;
  metadata?: Record<string, any>;
}

// WebSocket Message Types
export interface WSMessage {
  type: 'message' | 'tool_call' | 'tool_result' | 'ping' | 'pong' | 'error';
  content?: string;
  agent_role?: string;
  tool_name?: string;
  arguments?: Record<string, any>;
  success?: boolean;
  result?: any;
  error?: string;
  timestamp?: string;
  thinking?: string;
  task_id?: string;
}

// API Response Types
export interface APIResponse<T = any> {
  success?: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Health Check Types
export interface HealthStatus {
  status: string;
  components: {
    orchestrator: {
      active_agents: number;
      pending_tasks: number;
    };
    mcp_server: {
      registered_tools: number;
      active_sessions: number;
    };
    websocket: {
      active_connections: number;
    };
  };
  timestamp: string;
}

// Webhook Types
export interface Webhook {
  name: string;
  url: string;
  events: string[];
  active: boolean;
}

// Calendar Event Types
export interface CalendarEvent {
  id: string;
  title: string;
  start: string;
  end: string;
  description?: string;
}

// Application State Types
export interface AppState {
  // Connection
  connected: boolean;
  backendUrl: string;
  sessionId: string;

  // Agents
  agents: Agent[];
  selectedAgent: Agent | null;

  // Tasks
  tasks: Task[];
  activeTasks: Task[];

  // Chat
  messages: ChatMessage[];
  isTyping: boolean;

  // Tools
  mcpTools: MCPTool[];
  generatedTools: GeneratedTool[];

  // UI State
  sidebarOpen: boolean;
  activeView: 'chat' | 'agents' | 'planner' | 'tools' | 'editor';
  theme: 'light' | 'dark';
}

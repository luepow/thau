/**
 * THAU Code Desktop - API Service
 *
 * Handles REST API communication with THAU Code Server
 */

import axios, { AxiosInstance } from 'axios';
import {
  Agent,
  Task,
  Plan,
  MCPTool,
  MCPToolResult,
  GeneratedTool,
  HealthStatus,
  Webhook,
  CalendarEvent,
  TaskPriority,
  AgentRole,
} from '@/types';

class APIService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8001') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Update base URL (useful when connecting to remote server)
   */
  setBaseURL(url: string) {
    this.baseURL = url;
    this.client.defaults.baseURL = url;
  }

  /**
   * Health & Status
   */
  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  /**
   * Agents
   */
  async getAgents(): Promise<Agent[]> {
    const response = await this.client.get<{ agents: Agent[] }>('/api/agents');
    return response.data.agents;
  }

  async createTask(description: string, role: AgentRole, priority: TaskPriority = TaskPriority.MEDIUM): Promise<Task> {
    const response = await this.client.post<Task>('/api/agents/task', {
      description,
      role,
      priority,
    });
    return response.data;
  }

  async getTasks(): Promise<Task[]> {
    const response = await this.client.get<{ tasks: Task[] }>('/api/agents/tasks');
    return response.data.tasks;
  }

  async getTask(taskId: string): Promise<Task> {
    const response = await this.client.get<Task>(`/api/agents/tasks/${taskId}`);
    return response.data;
  }

  /**
   * Planner
   */
  async createPlan(taskDescription: string, priority: TaskPriority = TaskPriority.MEDIUM): Promise<Plan> {
    const response = await this.client.post<Plan>('/api/planner/create', {
      task_description: taskDescription,
      priority,
    });
    return response.data;
  }

  async analyzeTask(taskDescription: string): Promise<any> {
    const response = await this.client.post('/api/planner/analyze', null, {
      params: { task_description: taskDescription },
    });
    return response.data;
  }

  /**
   * MCP Tools
   */
  async getMCPTools(): Promise<MCPTool[]> {
    const response = await this.client.get<{ tools: MCPTool[] }>('/api/mcp/tools');
    return response.data.tools;
  }

  async invokeMCPTool(
    sessionId: string,
    toolName: string,
    args: Record<string, any>
  ): Promise<MCPToolResult> {
    const response = await this.client.post<MCPToolResult>('/api/mcp/invoke', {
      session_id: sessionId,
      tool_name: toolName,
      arguments: args,
    });
    return response.data;
  }

  async createMCPSession(sessionId?: string): Promise<any> {
    const response = await this.client.post('/api/mcp/session', null, {
      params: sessionId ? { session_id: sessionId } : {},
    });
    return response.data;
  }

  /**
   * Tool Factory
   */
  async createTool(description: string, templateName?: string): Promise<GeneratedTool> {
    const response = await this.client.post<GeneratedTool>('/api/tools/create', {
      description,
      template_name: templateName,
    });
    return response.data;
  }

  async getGeneratedTools(): Promise<GeneratedTool[]> {
    const response = await this.client.get<{ tools: GeneratedTool[] }>('/api/tools/list');
    return response.data.tools;
  }

  async getToolTemplates(): Promise<string[]> {
    const response = await this.client.get<{ templates: string[] }>('/api/tools/templates');
    return response.data.templates;
  }

  /**
   * API Toolkit
   */
  async getWebhooks(): Promise<Webhook[]> {
    const response = await this.client.get<{ webhooks: Webhook[] }>('/api/toolkit/webhooks');
    return response.data.webhooks;
  }

  async getCalendarEvents(): Promise<CalendarEvent[]> {
    const response = await this.client.get<{ events: CalendarEvent[] }>('/api/toolkit/calendar/events');
    return response.data.events;
  }
}

// Export singleton instance
export const apiService = new APIService();

export default APIService;

/**
 * THAU Code Desktop - Agent Panel Component
 */

import React, { useState, useEffect } from 'react';
import { Agent, Task, AgentRole } from '@/types';
import { apiService } from '@/services/api';

interface AgentPanelProps {
  selectedAgent: AgentRole;
  onAgentSelect: (agent: AgentRole) => void;
}

const AgentPanel: React.FC<AgentPanelProps> = ({
  selectedAgent,
  onAgentSelect,
}) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [agentsData, tasksData] = await Promise.all([
        apiService.getAgents(),
        apiService.getTasks(),
      ]);

      setAgents(agentsData);
      setTasks(tasksData);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load agents/tasks:', error);
      setLoading(false);
    }
  };

  const getAgentIcon = (role: AgentRole): string => {
    const icons: Record<AgentRole, string> = {
      [AgentRole.GENERAL]: '💬',
      [AgentRole.CODE_WRITER]: '✍️',
      [AgentRole.CODE_REVIEWER]: '👀',
      [AgentRole.DEBUGGER]: '🐛',
      [AgentRole.RESEARCHER]: '🔍',
      [AgentRole.PLANNER]: '📋',
      [AgentRole.ARCHITECT]: '🏗️',
      [AgentRole.TESTER]: '🧪',
      [AgentRole.DOCUMENTER]: '📝',
      [AgentRole.API_SPECIALIST]: '🔌',
      [AgentRole.DATA_ANALYST]: '📊',
      [AgentRole.SECURITY]: '🔒',
      [AgentRole.VISUAL_CREATOR]: '🎨',
    };
    return icons[role] || '🤖';
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', color: '#888' }}>
        Loading agents...
      </div>
    );
  }

  const activeTasks = tasks.filter(t => t.status === 'in_progress' || t.status === 'pending');

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div
        style={{
          padding: '16px',
          backgroundColor: '#252525',
          borderBottom: '1px solid #3e3e3e',
        }}
      >
        <h3 style={{ margin: 0, color: '#d4d4d4', fontSize: '16px' }}>
          THAU Agents
        </h3>
        <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
          {agents.length} agents | {activeTasks.length} active tasks
        </div>
      </div>

      {/* Agent List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
        {Object.values(AgentRole).map((role) => {
          const agent = agents.find(a => a.role === role);
          const agentTasks = tasks.filter(t => t.agent_role === role);
          const isSelected = selectedAgent === role;

          return (
            <div
              key={role}
              onClick={() => onAgentSelect(role)}
              style={{
                padding: '12px',
                margin: '4px 0',
                backgroundColor: isSelected ? '#007acc' : '#2d2d2d',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = '#3e3e3e';
                }
              }}
              onMouseLeave={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = '#2d2d2d';
                }
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>{getAgentIcon(role)}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ color: '#d4d4d4', fontSize: '14px', fontWeight: 'bold' }}>
                    {agent?.name || role.replace('_', ' ').toUpperCase()}
                  </div>
                  <div style={{ fontSize: '12px', color: '#aaa', marginTop: '2px' }}>
                    {agent?.description || 'Specialized agent'}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div
                    style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      backgroundColor: agent?.status === 'active' ? '#2d5a2d' : '#3e3e3e',
                      color: '#fff',
                    }}
                  >
                    {agent?.status || 'idle'}
                  </div>
                  {agentTasks.length > 0 && (
                    <div
                      style={{
                        fontSize: '10px',
                        marginTop: '4px',
                        color: '#888',
                      }}
                    >
                      {agentTasks.length} task{agentTasks.length > 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Active Tasks Summary */}
      {activeTasks.length > 0 && (
        <div
          style={{
            padding: '12px',
            backgroundColor: '#252525',
            borderTop: '1px solid #3e3e3e',
          }}
        >
          <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#d4d4d4', marginBottom: '8px' }}>
            Active Tasks ({activeTasks.length})
          </div>
          {activeTasks.slice(0, 3).map((task) => (
            <div
              key={task.id}
              style={{
                fontSize: '11px',
                color: '#aaa',
                padding: '4px 0',
                borderBottom: '1px solid #3e3e3e',
              }}
            >
              <div style={{ color: '#007acc' }}>{task.agent_role}</div>
              <div>{task.description.substring(0, 40)}...</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentPanel;

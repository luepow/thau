/**
 * THAU Code Desktop - Planner View Component
 */

import React, { useState } from 'react';
import { Plan, TaskPriority } from '@/types';
import { apiService } from '@/services/api';

const PlannerView: React.FC = () => {
  const [taskDescription, setTaskDescription] = useState('');
  const [priority, setPriority] = useState<TaskPriority>(TaskPriority.MEDIUM);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreatePlan = async () => {
    if (!taskDescription.trim()) {
      setError('Please enter a task description');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const newPlan = await apiService.createPlan(taskDescription, priority);
      setPlan(newPlan);
    } catch (err: any) {
      setError(err.message || 'Failed to create plan');
    } finally {
      setLoading(false);
    }
  };

  const getComplexityColor = (complexity: string): string => {
    const colors: Record<string, string> = {
      trivial: '#2d5a2d',
      low: '#4a7c4a',
      medium: '#7c7c4a',
      high: '#7c4a4a',
      very_high: '#8b2e2e',
    };
    return colors[complexity] || '#3e3e3e';
  };

  const getPriorityColor = (priority: string): string => {
    const colors: Record<string, string> = {
      low: '#3e3e3e',
      medium: '#7c7c4a',
      high: '#7c4a4a',
      critical: '#8b2e2e',
    };
    return colors[priority] || '#3e3e3e';
  };

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
          THAU Planner
        </h3>
        <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
          AI-powered task planning and decomposition
        </div>
      </div>

      {/* Input Section */}
      <div style={{ padding: '20px', backgroundColor: '#1e1e1e' }}>
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', color: '#d4d4d4', fontSize: '14px', marginBottom: '8px' }}>
            Task Description
          </label>
          <textarea
            value={taskDescription}
            onChange={(e) => setTaskDescription(e.target.value)}
            placeholder="Describe the task you want to plan..."
            style={{
              width: '100%',
              padding: '12px',
              backgroundColor: '#2d2d2d',
              color: '#d4d4d4',
              border: '1px solid #3e3e3e',
              borderRadius: '4px',
              fontSize: '14px',
              resize: 'vertical',
              minHeight: '100px',
              fontFamily: 'monospace',
            }}
          />
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', color: '#d4d4d4', fontSize: '14px', marginBottom: '8px' }}>
            Priority
          </label>
          <select
            value={priority}
            onChange={(e) => setPriority(e.target.value as TaskPriority)}
            style={{
              padding: '8px',
              backgroundColor: '#2d2d2d',
              color: '#d4d4d4',
              border: '1px solid #3e3e3e',
              borderRadius: '4px',
              fontSize: '14px',
              cursor: 'pointer',
            }}
          >
            <option value={TaskPriority.LOW}>Low</option>
            <option value={TaskPriority.MEDIUM}>Medium</option>
            <option value={TaskPriority.HIGH}>High</option>
            <option value={TaskPriority.CRITICAL}>Critical</option>
          </select>
        </div>

        <button
          onClick={handleCreatePlan}
          disabled={loading || !taskDescription.trim()}
          style={{
            padding: '12px 24px',
            backgroundColor: loading || !taskDescription.trim() ? '#3e3e3e' : '#007acc',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: loading || !taskDescription.trim() ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
          }}
        >
          {loading ? 'Creating Plan...' : 'Create Plan'}
        </button>

        {error && (
          <div style={{ marginTop: '12px', padding: '12px', backgroundColor: '#8b2e2e', borderRadius: '4px', color: '#fff', fontSize: '14px' }}>
            {error}
          </div>
        )}
      </div>

      {/* Plan Display */}
      {plan && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px', backgroundColor: '#1e1e1e' }}>
          {/* Plan Overview */}
          <div style={{ marginBottom: '20px', padding: '16px', backgroundColor: '#2d2d2d', borderRadius: '4px' }}>
            <h4 style={{ margin: '0 0 12px 0', color: '#d4d4d4' }}>Plan Overview</h4>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <div style={{ fontSize: '12px', color: '#888' }}>Complexity</div>
                <div
                  style={{
                    display: 'inline-block',
                    marginTop: '4px',
                    padding: '4px 8px',
                    backgroundColor: getComplexityColor(plan.complexity),
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                  }}
                >
                  {plan.complexity.toUpperCase()}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: '#888' }}>Priority</div>
                <div
                  style={{
                    display: 'inline-block',
                    marginTop: '4px',
                    padding: '4px 8px',
                    backgroundColor: getPriorityColor(plan.priority),
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                  }}
                >
                  {plan.priority.toUpperCase()}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: '#888' }}>Estimated Hours</div>
                <div style={{ fontSize: '18px', color: '#d4d4d4', marginTop: '4px' }}>
                  {plan.estimated_hours}h
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: '#888' }}>Steps</div>
                <div style={{ fontSize: '18px', color: '#d4d4d4', marginTop: '4px' }}>
                  {plan.steps.length}
                </div>
              </div>
            </div>
          </div>

          {/* Steps */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#d4d4d4', marginBottom: '12px' }}>Implementation Steps</h4>

            {plan.steps.map((step) => (
              <div
                key={step.step_number}
                style={{
                  marginBottom: '12px',
                  padding: '12px',
                  backgroundColor: '#2d2d2d',
                  borderRadius: '4px',
                  borderLeft: '3px solid #007acc',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                  <div style={{ fontWeight: 'bold', color: '#007acc' }}>
                    Step {step.step_number}
                  </div>
                  <div
                    style={{
                      fontSize: '11px',
                      padding: '2px 6px',
                      backgroundColor: '#3e3e3e',
                      borderRadius: '3px',
                    }}
                  >
                    {step.action_type}
                  </div>
                </div>

                <div style={{ color: '#d4d4d4', marginBottom: '8px' }}>
                  {step.description}
                </div>

                <div style={{ display: 'flex', gap: '12px', fontSize: '12px', color: '#888' }}>
                  <div>Effort: <strong>{step.estimated_effort}</strong></div>
                  {step.dependencies.length > 0 && (
                    <div>Dependencies: <strong>Steps {step.dependencies.join(', ')}</strong></div>
                  )}
                </div>

                {step.tools_needed.length > 0 && (
                  <div style={{ marginTop: '8px', fontSize: '12px' }}>
                    <div style={{ color: '#888', marginBottom: '4px' }}>Tools needed:</div>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {step.tools_needed.map((tool, index) => (
                        <span
                          key={index}
                          style={{
                            padding: '2px 8px',
                            backgroundColor: '#3e3e3e',
                            borderRadius: '3px',
                            fontSize: '11px',
                            color: '#aaa',
                          }}
                        >
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Risks & Assumptions */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            {plan.risks.length > 0 && (
              <div>
                <h4 style={{ color: '#d4d4d4', marginBottom: '12px' }}>Risks</h4>
                <ul style={{ margin: 0, paddingLeft: '20px', color: '#d4d4d4', fontSize: '14px' }}>
                  {plan.risks.map((risk, index) => (
                    <li key={index} style={{ marginBottom: '8px' }}>{risk}</li>
                  ))}
                </ul>
              </div>
            )}

            {plan.assumptions.length > 0 && (
              <div>
                <h4 style={{ color: '#d4d4d4', marginBottom: '12px' }}>Assumptions</h4>
                <ul style={{ margin: 0, paddingLeft: '20px', color: '#d4d4d4', fontSize: '14px' }}>
                  {plan.assumptions.map((assumption, index) => (
                    <li key={index} style={{ marginBottom: '8px' }}>{assumption}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Success Criteria */}
          {plan.success_criteria.length > 0 && (
            <div style={{ marginTop: '16px' }}>
              <h4 style={{ color: '#d4d4d4', marginBottom: '12px' }}>Success Criteria</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', color: '#d4d4d4', fontSize: '14px' }}>
                {plan.success_criteria.map((criteria, index) => (
                  <li key={index} style={{ marginBottom: '8px' }}>{criteria}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PlannerView;

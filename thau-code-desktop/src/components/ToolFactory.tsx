/**
 * THAU Code Desktop - Tool Factory UI Component
 */

import React, { useState, useEffect } from 'react';
import { GeneratedTool, MCPTool } from '@/types';
import { apiService } from '@/services/api';

const ToolFactory: React.FC = () => {
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedTools, setGeneratedTools] = useState<GeneratedTool[]>([]);
  const [mcpTools, setMCPTools] = useState<MCPTool[]>([]);
  const [activeTab, setActiveTab] = useState<'create' | 'generated' | 'mcp'>('create');

  useEffect(() => {
    loadTools();
  }, []);

  const loadTools = async () => {
    try {
      const [generated, mcp] = await Promise.all([
        apiService.getGeneratedTools(),
        apiService.getMCPTools(),
      ]);

      setGeneratedTools(generated);
      setMCPTools(mcp);
    } catch (err: any) {
      console.error('Failed to load tools:', err);
    }
  };

  const handleCreateTool = async () => {
    if (!description.trim()) {
      setError('Please enter a tool description');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const tool = await apiService.createTool(description);
      setGeneratedTools([tool, ...generatedTools]);
      setDescription('');
      setActiveTab('generated');
    } catch (err: any) {
      setError(err.message || 'Failed to create tool');
    } finally {
      setLoading(false);
    }
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
          THAU Tool Factory
        </h3>
        <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
          Auto-generate tools from natural language descriptions
        </div>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          padding: '8px',
          backgroundColor: '#252525',
          borderBottom: '1px solid #3e3e3e',
        }}
      >
        {['create', 'generated', 'mcp'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            style={{
              padding: '8px 16px',
              backgroundColor: activeTab === tab ? '#007acc' : '#2d2d2d',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
            }}
          >
            {tab === 'create' && `Create (${generatedTools.length})`}
            {tab === 'generated' && `Generated (${generatedTools.length})`}
            {tab === 'mcp' && `MCP Tools (${mcpTools.length})`}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {activeTab === 'create' && (
          <div style={{ padding: '20px' }}>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', color: '#d4d4d4', fontSize: '14px', marginBottom: '8px' }}>
                Tool Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe the tool you want to create...&#10;&#10;Examples:&#10;- 'Send email notifications with HTML templates'&#10;- 'Fetch weather data from OpenWeatherMap API'&#10;- 'Convert PDF documents to text'"
                style={{
                  width: '100%',
                  padding: '12px',
                  backgroundColor: '#2d2d2d',
                  color: '#d4d4d4',
                  border: '1px solid #3e3e3e',
                  borderRadius: '4px',
                  fontSize: '14px',
                  resize: 'vertical',
                  minHeight: '150px',
                  fontFamily: 'monospace',
                }}
              />
            </div>

            <button
              onClick={handleCreateTool}
              disabled={loading || !description.trim()}
              style={{
                padding: '12px 24px',
                backgroundColor: loading || !description.trim() ? '#3e3e3e' : '#007acc',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: loading || !description.trim() ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: 'bold',
              }}
            >
              {loading ? 'Generating Tool...' : 'Generate Tool'}
            </button>

            {error && (
              <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#8b2e2e', borderRadius: '4px', color: '#fff', fontSize: '14px' }}>
                {error}
              </div>
            )}

            {/* Examples */}
            <div style={{ marginTop: '32px', padding: '16px', backgroundColor: '#2d2d2d', borderRadius: '4px' }}>
              <h4 style={{ margin: '0 0 12px 0', color: '#d4d4d4', fontSize: '14px' }}>
                Tool Creation Examples
              </h4>
              <div style={{ fontSize: '12px', color: '#aaa', lineHeight: '1.6' }}>
                <strong style={{ color: '#007acc' }}>API Tools:</strong>
                <div style={{ marginLeft: '12px', marginTop: '4px' }}>
                  "Call REST API to get user information with pagination"
                </div>

                <strong style={{ color: '#007acc', marginTop: '12px', display: 'block' }}>Calendar Tools:</strong>
                <div style={{ marginLeft: '12px', marginTop: '4px' }}>
                  "Create calendar events with timezone support"
                </div>

                <strong style={{ color: '#007acc', marginTop: '12px', display: 'block' }}>Database Tools:</strong>
                <div style={{ marginLeft: '12px', marginTop: '4px' }}>
                  "Query PostgreSQL database with parameterized queries"
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'generated' && (
          <div style={{ padding: '20px' }}>
            {generatedTools.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
                No tools generated yet. Switch to 'Create' tab to generate your first tool!
              </div>
            ) : (
              generatedTools.map((tool) => (
                <div
                  key={tool.name}
                  style={{
                    marginBottom: '16px',
                    padding: '16px',
                    backgroundColor: '#2d2d2d',
                    borderRadius: '4px',
                    borderLeft: '3px solid #007acc',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                    <h4 style={{ margin: 0, color: '#d4d4d4', fontSize: '16px' }}>
                      {tool.name}
                    </h4>
                    <div
                      style={{
                        fontSize: '11px',
                        padding: '2px 8px',
                        backgroundColor: '#3e3e3e',
                        borderRadius: '3px',
                        color: '#aaa',
                      }}
                    >
                      {tool.category}
                    </div>
                  </div>

                  <div style={{ color: '#aaa', fontSize: '14px', marginBottom: '12px' }}>
                    {tool.description}
                  </div>

                  {tool.parameters.length > 0 && (
                    <div>
                      <div style={{ fontSize: '12px', color: '#888', marginBottom: '8px' }}>
                        Parameters:
                      </div>
                      {tool.parameters.map((param) => (
                        <div
                          key={param.name}
                          style={{
                            fontSize: '12px',
                            color: '#aaa',
                            marginLeft: '12px',
                            marginBottom: '4px',
                          }}
                        >
                          <strong style={{ color: '#007acc' }}>{param.name}</strong>
                          {' '}({param.type}){param.required && ' *'} - {param.description}
                        </div>
                      ))}
                    </div>
                  )}

                  {tool.file_path && (
                    <div style={{ marginTop: '12px', fontSize: '11px', color: '#666' }}>
                      File: {tool.file_path}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'mcp' && (
          <div style={{ padding: '20px' }}>
            {mcpTools.map((tool) => (
              <div
                key={tool.function.name}
                style={{
                  marginBottom: '16px',
                  padding: '16px',
                  backgroundColor: '#2d2d2d',
                  borderRadius: '4px',
                  borderLeft: '3px solid #2d5a2d',
                }}
              >
                <h4 style={{ margin: '0 0 8px 0', color: '#d4d4d4', fontSize: '16px' }}>
                  {tool.function.name}
                </h4>

                <div style={{ color: '#aaa', fontSize: '14px', marginBottom: '12px' }}>
                  {tool.function.description}
                </div>

                {Object.keys(tool.function.parameters.properties).length > 0 && (
                  <div>
                    <div style={{ fontSize: '12px', color: '#888', marginBottom: '8px' }}>
                      Parameters:
                    </div>
                    {Object.entries(tool.function.parameters.properties).map(([name, info]: [string, any]) => (
                      <div
                        key={name}
                        style={{
                          fontSize: '12px',
                          color: '#aaa',
                          marginLeft: '12px',
                          marginBottom: '4px',
                        }}
                      >
                        <strong style={{ color: '#2d7a2d' }}>{name}</strong>
                        {' '}({info.type}){tool.function.parameters.required.includes(name) && ' *'} - {info.description}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ToolFactory;

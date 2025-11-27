const React = require('react');
const { Box, Text, useInput, useApp } = require('ink');
const TextInput = require('ink-text-input').default;
const chalk = require('chalk');

/**
 * Multi-Console UI Component
 * Displays multiple agent sessions as tabs
 */
const MultiConsole = ({ sessionManager, thauClient, onExit }) => {
  const { exit } = useApp();
  const [activeInput, setActiveInput] = React.useState('');
  const [inputMode, setInputMode] = React.useState('chat'); // 'chat' | 'command'
  const [sessions, setSessions] = React.useState([]);
  const [activeSessionId, setActiveSessionId] = React.useState(null);
  const [showHelp, setShowHelp] = React.useState(false);
  const [isProcessing, setIsProcessing] = React.useState(false);

  // Update sessions when they change
  React.useEffect(() => {
    const updateSessions = () => {
      setSessions(sessionManager.listSessions());
      setActiveSessionId(sessionManager.activeSessionId);
    };

    // Initial load
    updateSessions();

    // Subscribe to session events
    sessionManager.on('session:created', updateSessions);
    sessionManager.on('session:deleted', updateSessions);
    sessionManager.on('session:switched', updateSessions);
    sessionManager.on('message:added', updateSessions);

    return () => {
      sessionManager.removeListener('session:created', updateSessions);
      sessionManager.removeListener('session:deleted', updateSessions);
      sessionManager.removeListener('session:switched', updateSessions);
      sessionManager.removeListener('message:added', updateSessions);
    };
  }, [sessionManager]);

  // Handle keyboard input
  useInput((input, key) => {
    // Ctrl+C to exit
    if (key.ctrl && input === 'c') {
      if (onExit) onExit();
      exit();
      return;
    }

    // Ctrl+H to toggle help
    if (key.ctrl && input === 'h') {
      setShowHelp(!showHelp);
      return;
    }

    // Ctrl+N to create new session
    if (key.ctrl && input === 'n') {
      const agentName = `agent-${sessions.length + 1}`;
      const sessionId = sessionManager.createSession(agentName);
      setActiveSessionId(sessionId);
      return;
    }

    // Ctrl+W to close current session
    if (key.ctrl && input === 'w') {
      if (sessions.length > 1) {
        sessionManager.deleteSession(activeSessionId);
      }
      return;
    }

    // Ctrl+Tab to switch to next session
    if (key.ctrl && key.tab) {
      const currentIndex = sessions.findIndex(s => s.id === activeSessionId);
      const nextIndex = (currentIndex + 1) % sessions.length;
      if (sessions[nextIndex]) {
        sessionManager.switchSession(sessions[nextIndex].id);
      }
      return;
    }

    // Alt+[1-9] to switch to specific session
    if (key.meta && input >= '1' && input <= '9') {
      const index = parseInt(input) - 1;
      if (sessions[index]) {
        sessionManager.switchSession(sessions[index].id);
      }
      return;
    }
  });

  const handleSubmit = async (value) => {
    if (!value.trim() || isProcessing) return;

    const currentSession = sessionManager.getActiveSession();
    if (!currentSession) return;

    // Check if it's a command
    if (value.startsWith('/')) {
      await handleCommand(value, currentSession);
      setActiveInput('');
      return;
    }

    // Add user message
    sessionManager.addMessage(currentSession.id, 'user', value);
    setActiveInput('');
    setIsProcessing(true);

    try {
      // Get conversation history for context
      const history = sessionManager.getHistory(currentSession.id, 5);
      const contextMessages = history
        .slice(0, -1) // Exclude the just-added message
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));

      // Send to THAU API or Ollama
      const response = await thauClient.sendTask(
        value,
        currentSession.agent,
        contextMessages
      );

      const answer = response.result || response.description || 'No response';

      // Add assistant response
      sessionManager.addMessage(
        currentSession.id,
        'assistant',
        answer
      );
    } catch (error) {
      // Add error message
      sessionManager.addMessage(
        currentSession.id,
        'system',
        `Error: ${error.message}`
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCommand = async (command, session) => {
    const parts = command.split(' ');
    const cmd = parts[0];

    switch (cmd) {
      case '/help':
        setShowHelp(true);
        break;

      case '/agent':
        if (parts[1]) {
          sessionManager.changeAgent(session.id, parts[1]);
          sessionManager.addMessage(
            session.id,
            'system',
            `Agent changed to: ${parts[1]}`
          );
        }
        break;

      case '/model':
        if (parts[1]) {
          sessionManager.changeModel(session.id, parts[1]);
          sessionManager.addMessage(
            session.id,
            'system',
            `Model changed to: ${parts[1]}`
          );
        }
        break;

      case '/clear':
        sessionManager.clearHistory(session.id);
        break;

      case '/new':
        const agentName = parts[1] || 'general';
        const newSessionId = sessionManager.createSession(agentName);
        sessionManager.switchSession(newSessionId);
        break;

      case '/close':
        if (sessions.length > 1) {
          sessionManager.deleteSession(session.id);
        }
        break;

      case '/exit':
        if (onExit) onExit();
        exit();
        break;

      default:
        sessionManager.addMessage(
          session.id,
          'system',
          `Unknown command: ${cmd}. Type /help for available commands.`
        );
    }
  };

  const activeSession = sessions.find(s => s.id === activeSessionId);
  const history = activeSession ? sessionManager.getHistory(activeSession.id, 20) : [];

  if (showHelp) {
    return <HelpScreen onClose={() => setShowHelp(false)} />;
  }

  return (
    <Box flexDirection="column" height="100%">
      {/* Header with tabs */}
      <Box borderStyle="single" borderColor="cyan" paddingX={1}>
        <Text bold color="cyan">THAU Multi-Console</Text>
        <Text dimColor> - </Text>
        {sessions.map((session, idx) => (
          <React.Fragment key={session.id}>
            <Text
              bold={session.isActive}
              color={session.isActive ? 'yellow' : 'gray'}
              backgroundColor={session.isActive ? 'blue' : undefined}
            >
              [{idx + 1}] {session.agent}
            </Text>
            {idx < sessions.length - 1 && <Text dimColor> | </Text>}
          </React.Fragment>
        ))}
      </Box>

      {/* Active session info */}
      {activeSession && (
        <Box borderStyle="single" borderColor="gray" paddingX={1}>
          <Text>
            <Text bold color="green">Agent:</Text> {activeSession.agent} {' | '}
            <Text bold color="green">Model:</Text> {activeSession.model} {' | '}
            <Text bold color="green">Messages:</Text> {activeSession.messageCount}
          </Text>
        </Box>
      )}

      {/* Conversation history */}
      <Box flexDirection="column" flexGrow={1} paddingX={1} paddingY={1}>
        {history.map((msg, idx) => (
          <Message key={idx} message={msg} />
        ))}
      </Box>

      {/* Input area */}
      <Box borderStyle="single" borderColor="cyan" paddingX={1}>
        <Text bold color="cyan">{activeSession ? activeSession.agent : 'No session'}&gt; </Text>
        <TextInput
          value={activeInput}
          onChange={setActiveInput}
          onSubmit={handleSubmit}
          placeholder={isProcessing ? "Processing..." : "Type your message or /help for commands..."}
          isDisabled={isProcessing}
        />
      </Box>

      {/* Status bar */}
      <Box borderStyle="single" borderColor="gray" paddingX={1}>
        {isProcessing ? (
          <Text color="yellow">⏳ Processing request...</Text>
        ) : (
          <Text dimColor>
            Ctrl+N: New | Ctrl+W: Close | Ctrl+Tab: Next | Alt+[1-9]: Switch | Ctrl+H: Help | Ctrl+C: Exit
          </Text>
        )}
      </Box>
    </Box>
  );
};

/**
 * Message component
 */
const Message = ({ message }) => {
  const { role, content, timestamp } = message;

  let color = 'white';
  let prefix = '';

  switch (role) {
    case 'user':
      color = 'cyan';
      prefix = 'You';
      break;
    case 'assistant':
      color = 'magenta';
      prefix = 'THAU';
      break;
    case 'system':
      color = 'yellow';
      prefix = 'System';
      break;
  }

  const time = timestamp ? new Date(timestamp).toLocaleTimeString() : '';

  return (
    <Box flexDirection="column" marginY={0}>
      <Box>
        <Text bold color={color}>{prefix}</Text>
        <Text dimColor> [{time}]</Text>
      </Box>
      <Box paddingLeft={2}>
        <Text>{content}</Text>
      </Box>
    </Box>
  );
};

/**
 * Help screen component
 */
const HelpScreen = ({ onClose }) => {
  useInput((input, key) => {
    if (input === 'q' || (key.ctrl && input === 'h') || key.escape) {
      onClose();
    }
  });

  return (
    <Box flexDirection="column" borderStyle="double" borderColor="cyan" padding={2}>
      <Text bold color="cyan">THAU Multi-Console - Help</Text>
      <Text> </Text>

      <Text bold color="yellow">Keyboard Shortcuts:</Text>
      <Box flexDirection="column" paddingLeft={2}>
        <Text><Text color="green">Ctrl+N</Text>     - Create new session</Text>
        <Text><Text color="green">Ctrl+W</Text>     - Close current session</Text>
        <Text><Text color="green">Ctrl+Tab</Text>   - Switch to next session</Text>
        <Text><Text color="green">Alt+[1-9]</Text>  - Switch to session 1-9</Text>
        <Text><Text color="green">Ctrl+H</Text>     - Toggle this help screen</Text>
        <Text><Text color="green">Ctrl+C</Text>     - Exit THAU</Text>
      </Box>

      <Text> </Text>
      <Text bold color="yellow">Commands:</Text>
      <Box flexDirection="column" paddingLeft={2}>
        <Text><Text color="green">/help</Text>              - Show this help</Text>
        <Text><Text color="green">/agent &lt;name&gt;</Text>     - Change agent for current session</Text>
        <Text><Text color="green">/model &lt;name&gt;</Text>     - Change model for current session</Text>
        <Text><Text color="green">/new [agent]</Text>       - Create new session with agent</Text>
        <Text><Text color="green">/close</Text>             - Close current session</Text>
        <Text><Text color="green">/clear</Text>             - Clear current session history</Text>
        <Text><Text color="green">/exit</Text>              - Exit THAU</Text>
      </Box>

      <Text> </Text>
      <Text bold color="yellow">Available Agents:</Text>
      <Box flexDirection="column" paddingLeft={2}>
        <Text>general, code_writer, planner, code_reviewer, debugger,</Text>
        <Text>architect, test_writer, refactorer, explainer, optimizer, security</Text>
      </Box>

      <Text> </Text>
      <Text dimColor>Press 'q' or ESC to close help</Text>
    </Box>
  );
};

module.exports = MultiConsole;

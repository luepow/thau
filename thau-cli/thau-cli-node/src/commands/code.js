const inquirer = require('inquirer');
const chalk = require('chalk');
const marked = require('marked');
const { markedTerminal } = require('marked-terminal');
const ThauClient = require('../lib/client');
const { spawn } = require('child_process');

marked.use(markedTerminal());

const AGENTS = [
  'general',
  'code_writer',
  'planner',
  'code_reviewer',
  'debugger',
  'architect',
  'test_writer',
  'refactorer',
  'explainer',
  'optimizer',
  'security'
];

async function handleCommand(message, currentAgent, conversationHistory, client) {
  const parts = message.split(' ');
  const cmd = parts[0];

  if (cmd === '/help') {
    console.log(chalk.cyan(`
Available commands:
  /help            - Show this help message
  /agent <name>    - Switch agent (${AGENTS.join(', ')})
  /model           - List and switch models (THAU API / Ollama)
  /mcp             - MCP server management
  /permissions     - View/manage permissions
  /exec <command>  - Execute shell command (requires permission)
  /clear           - Clear conversation history
  /exit            - Exit interactive mode
    `));
    return 'continue';
  }

  if (cmd === '/agent') {
    const agentName = parts[1];
    if (!agentName) {
      console.log(chalk.yellow('Usage: /agent <name>'));
      console.log(chalk.gray('Available agents: ' + AGENTS.join(', ')));
      return 'continue';
    }
    if (!AGENTS.includes(agentName)) {
      console.log(chalk.red(`Unknown agent: ${agentName}`));
      console.log(chalk.gray('Available agents: ' + AGENTS.join(', ')));
      return 'continue';
    }
    console.log(chalk.green(`Agent switched to: ${agentName}`));
    return agentName;
  }

  if (cmd === '/model') {
    const subCmd = parts[1];

    if (!subCmd || subCmd === 'list') {
      // List available models
      console.log(chalk.cyan('\n📦 Available Models:\n'));
      try {
        const models = await client.listAvailableModels();
        models.forEach((model, idx) => {
          const available = model.available ? chalk.green('✓') : chalk.red('✗');
          const current = model.name === client.currentModel ||
                         model.name === `ollama:${client.currentOllamaModel}` ?
                         chalk.yellow(' (current)') : '';
          console.log(`  ${available} ${chalk.bold(model.name)} ${chalk.gray(`[${model.type}]`)}${current}`);
          if (model.size) {
            console.log(chalk.gray(`     Size: ${(model.size / 1e9).toFixed(2)} GB`));
          }
        });
      } catch (error) {
        console.log(chalk.red('Error listing models: ' + error.message));
      }
      return 'continue';
    }

    if (subCmd === 'switch') {
      const modelName = parts.slice(2).join(' ');
      if (!modelName) {
        console.log(chalk.yellow('Usage: /model switch <model-name>'));
        console.log(chalk.gray('Example: /model switch ollama:codellama'));
        console.log(chalk.gray('Example: /model switch thau-api'));
        return 'continue';
      }

      try {
        if (modelName.startsWith('ollama:')) {
          const ollamaModel = modelName.replace('ollama:', '');
          await client.switchModel('ollama', ollamaModel);
          console.log(chalk.green(`✓ Switched to Ollama model: ${ollamaModel}`));
        } else if (modelName === 'thau-api') {
          await client.switchModel('thau-api');
          console.log(chalk.green('✓ Switched to THAU API'));
        } else {
          console.log(chalk.red(`Unknown model: ${modelName}`));
        }
      } catch (error) {
        console.log(chalk.red('Error switching model: ' + error.message));
      }
      return 'continue';
    }

    console.log(chalk.yellow('Usage: /model [list|switch <name>]'));
    return 'continue';
  }

  if (cmd === '/mcp') {
    const subCmd = parts[1];

    if (!subCmd || subCmd === 'status') {
      client.mcpManager.showStatus();
      return 'continue';
    }

    if (subCmd === 'connect') {
      const serverName = parts[2];
      if (!serverName) {
        console.log(chalk.yellow('Usage: /mcp connect <server-name>'));
        return 'continue';
      }

      try {
        await client.mcpManager.connect(serverName);
        console.log(chalk.green(`✓ Connected to ${serverName}`));
      } catch (error) {
        console.log(chalk.red('Error: ' + error.message));
      }
      return 'continue';
    }

    if (subCmd === 'disconnect') {
      const serverName = parts[2];
      if (!serverName) {
        console.log(chalk.yellow('Usage: /mcp disconnect <server-name>'));
        return 'continue';
      }

      const success = client.mcpManager.disconnect(serverName);
      if (success) {
        console.log(chalk.green(`✓ Disconnected from ${serverName}`));
      } else {
        console.log(chalk.yellow(`Not connected to ${serverName}`));
      }
      return 'continue';
    }

    if (subCmd === 'add') {
      const serverName = parts[2];
      const serverUrl = parts[3];
      if (!serverName || !serverUrl) {
        console.log(chalk.yellow('Usage: /mcp add <name> <url>'));
        console.log(chalk.gray('Example: /mcp add my-server http://localhost:9000'));
        return 'continue';
      }

      try {
        client.mcpManager.addServer(serverName, serverUrl);
        console.log(chalk.green(`✓ Added MCP server: ${serverName}`));
      } catch (error) {
        console.log(chalk.red('Error: ' + error.message));
      }
      return 'continue';
    }

    if (subCmd === 'tools') {
      const serverName = parts[2];
      if (!serverName) {
        console.log(chalk.yellow('Usage: /mcp tools <server-name>'));
        return 'continue';
      }

      try {
        const tools = await client.mcpManager.listTools(serverName);
        console.log(chalk.cyan(`\n🔧 Available Tools on ${serverName}:\n`));
        tools.forEach(tool => {
          console.log(`  ${chalk.bold(tool.name)}`);
          if (tool.description) {
            console.log(chalk.gray(`     ${tool.description}`));
          }
        });
      } catch (error) {
        console.log(chalk.red('Error: ' + error.message));
      }
      return 'continue';
    }

    console.log(chalk.yellow('Usage: /mcp [status|connect|disconnect|add|tools]'));
    return 'continue';
  }

  if (cmd === '/permissions') {
    const subCmd = parts[1];

    if (!subCmd || subCmd === 'show') {
      client.permissions.showPermissions();
      return 'continue';
    }

    if (subCmd === 'reset') {
      const { confirm } = await inquirer.prompt([{
        type: 'confirm',
        name: 'confirm',
        message: 'Reset all permissions to default?',
        default: false
      }]);

      if (confirm) {
        client.permissions.resetPermissions();
        console.log(chalk.green('✓ Permissions reset'));
      }
      return 'continue';
    }

    console.log(chalk.yellow('Usage: /permissions [show|reset]'));
    return 'continue';
  }

  if (cmd === '/exec') {
    const command = parts.slice(1).join(' ');
    if (!command) {
      console.log(chalk.yellow('Usage: /exec <command>'));
      console.log(chalk.gray('Example: /exec ls -la'));
      return 'continue';
    }

    try {
      const allowed = await client.permissions.requestCommandExecution(command);
      if (!allowed) {
        console.log(chalk.red('✗ Permission denied'));
        return 'continue';
      }

      console.log(chalk.gray(`\n$ ${command}\n`));

      const child = spawn(command, [], {
        shell: true,
        cwd: process.cwd()
      });

      child.stdout.on('data', (data) => {
        process.stdout.write(data.toString());
      });

      child.stderr.on('data', (data) => {
        process.stderr.write(chalk.red(data.toString()));
      });

      await new Promise((resolve) => {
        child.on('close', (code) => {
          if (code !== 0) {
            console.log(chalk.red(`\nExited with code ${code}`));
          }
          resolve();
        });
      });

    } catch (error) {
      console.log(chalk.red('Error executing command: ' + error.message));
    }
    return 'continue';
  }

  if (cmd === '/clear') {
    conversationHistory.length = 0;
    console.log(chalk.green('Conversation history cleared'));
    return 'continue';
  }

  if (cmd === '/exit') {
    return 'exit';
  }

  console.log(chalk.yellow(`Unknown command: ${cmd}`));
  console.log(chalk.gray('Type /help for available commands'));
  return 'continue';
}

async function codeCommand() {
  console.log(chalk.cyan.bold('\n💻 THAU Interactive Coding Mode\n'));

  const client = new ThauClient();

  // Auto-initialize .thau/ project config if not exists
  if (!client.projectInit.isInitialized()) {
    console.log(chalk.gray('Initializing project...'));
    await client.projectInit.initialize();
    console.log(chalk.green('✓ Project initialized\n'));
  }

  // Check server health and auto-switch to Ollama if needed
  const healthy = await client.healthCheck();
  if (!healthy && client.currentModel === 'thau-api') {
    const ollamaAvailable = await client.ollamaClient.isAvailable();
    if (ollamaAvailable) {
      console.log(chalk.yellow('⚠️  THAU server unavailable, switching to Ollama...'));
      await client.switchModel('ollama');
      console.log(chalk.green('✓ Connected to Ollama\n'));
    } else {
      console.log(chalk.red('❌ Neither THAU server nor Ollama are available.'));
      console.log(chalk.yellow('Please start THAU server or install Ollama.'));
      console.log(chalk.gray('THAU Server: python api/thau_code_server.py'));
      console.log(chalk.gray('Ollama: https://ollama.ai\n'));
      return;
    }
  } else {
    const modelDisplay = client.currentModel === 'ollama' ?
      `Ollama (${client.currentOllamaModel})` :
      'THAU API';
    console.log(chalk.green(`✓ Connected to: ${modelDisplay}\n`));
  }

  let currentAgent = client.config.default_agent || 'code_writer';
  const conversationHistory = [];

  console.log(chalk.gray(`Agent: ${currentAgent} | Type /help for commands\n`));

  while (true) {
    const { message } = await inquirer.prompt([{
      type: 'input',
      name: 'message',
      message: chalk.cyan('You:'),
    }]);

    if (!message.trim()) {
      continue;
    }

    // Handle slash commands
    if (message.startsWith('/')) {
      const result = await handleCommand(message, currentAgent, conversationHistory, client);
      if (result === 'exit') {
        console.log(chalk.yellow('\n👋 Goodbye!\n'));
        break;
      }
      if (result === 'continue') {
        continue;
      }
      // Agent switched
      currentAgent = result;
      continue;
    }

    // Add user message to history
    conversationHistory.push({
      role: 'user',
      content: message
    });

    try {
      // Send to THAU
      const response = await client.sendTask(
        message,
        currentAgent,
        conversationHistory.slice(-5) // Last 5 messages for context
      );

      const answer = response.result || response.description || 'No response';

      // Add assistant response to history
      conversationHistory.push({
        role: 'assistant',
        content: answer
      });

      console.log(chalk.magenta('\nTHAU:'));

      // Render markdown if present
      if (answer.includes('```') || answer.includes('#')) {
        console.log(marked(answer));
      } else {
        console.log(answer);
      }

      console.log();

    } catch (error) {
      console.log(chalk.red('Error:'), error.message);
    }
  }
}

module.exports = codeCommand;

const inquirer = require('inquirer');
const chalk = require('chalk');
const marked = require('marked');
const { markedTerminal } = require('marked-terminal');
const boxen = require('boxen');
const ThauClient = require('../lib/client');
const { spawn } = require('child_process');

marked.use(markedTerminal());

async function codeCommand() {
  const client = new ThauClient();

  // Check server
  const isHealthy = await client.healthCheck();
  if (!isHealthy) {
    console.log(chalk.red('❌ Error: THAU server not running'));
    console.log(chalk.yellow('Start server with:'), 'python api/thau_code_server.py');
    process.exit(1);
  }

  // Welcome
  console.log(
    boxen(
      chalk.cyan.bold('🧠 THAU CODE - Interactive Mode\n\n') +
        chalk.dim('Like Claude Code, but powered by THAU AI\n\n') +
        chalk.cyan('Agent: ') +
        client.config.default_agent +
        '\n' +
        chalk.cyan('Server: ') +
        client.config.server_url +
        '\n\n' +
        chalk.dim('Commands:\n') +
        '  /help    - Show help\n' +
        '  /agent   - Switch agent\n' +
        '  /clear   - Clear history\n' +
        '  /exit    - Exit',
      {
        padding: 1,
        margin: 1,
        borderStyle: 'round',
        borderColor: 'cyan',
      }
    )
  );

  const conversationHistory = [];
  let currentAgent = client.config.default_agent;

  // Interactive loop
  while (true) {
    const { message } = await inquirer.prompt([
      {
        type: 'input',
        name: 'message',
        message: chalk.cyan('You:'),
      },
    ]);

    if (!message.trim()) continue;

    // Handle commands
    if (message.startsWith('/')) {
      const handled = await handleCommand(message, currentAgent, conversationHistory, client);
      if (handled === 'exit') break;
      if (handled === 'agent') {
        const newAgent = await switchAgent();
        if (newAgent) currentAgent = newAgent;
      }
      continue;
    }

    // Send to THAU
    conversationHistory.push({ role: 'user', content: message });

    console.log(chalk.cyan('\n🧠 ' + currentAgent + ' thinking...\n'));

    const response = await client.sendTask(
      message,
      currentAgent,
      conversationHistory.slice(-5)
    );

    if (response.error) {
      console.log(chalk.red('Error:'), response.error);
    } else {
      const answer = response.result || response.description || 'No response';
      conversationHistory.push({ role: 'assistant', content: answer });

      console.log(chalk.green('\n🤖 ' + currentAgent + ':\n'));

      // Render markdown if it looks like markdown
      if (answer.includes('```') || answer.includes('#') || answer.includes('*')) {
        console.log(marked(answer));
      } else {
        console.log(answer);
      }
      console.log();
    }
  }

  console.log(chalk.cyan('\n👋 Goodbye!\n'));
}

async function handleCommand(command, currentAgent, history, client) {
  const cmd = command.toLowerCase().trim();

  if (cmd === '/exit' || cmd === '/quit') {
    return 'exit';
  }

  if (cmd === '/help') {
    console.log(
      boxen(
        chalk.bold('THAU CODE Commands\n\n') +
          chalk.cyan('/help') +
          '     - Show this help\n' +
          chalk.cyan('/agent') +
          '    - Switch agent\n' +
          chalk.cyan('/clear') +
          '    - Clear conversation history\n' +
          chalk.cyan('/exit') +
          '     - Exit interactive mode\n\n' +
          chalk.bold('Usage Examples:\n') +
          '  Create a Python function to calculate fibonacci\n' +
          '  Refactor this code to use dependency injection\n' +
          '  Plan a REST API with authentication',
        {
          padding: 1,
          borderStyle: 'round',
          borderColor: 'blue',
          title: '💡 Help',
        }
      )
    );
    return true;
  }

  if (cmd === '/clear') {
    history.length = 0;
    console.log(chalk.green('✓ Conversation history cleared'));
    return true;
  }

  if (cmd === '/agent') {
    return 'agent';
  }

  console.log(chalk.red('Unknown command:'), command);
  console.log(chalk.dim('Type /help for available commands'));
  return true;
}

async function switchAgent() {
  const agents = [
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
    'security',
  ];

  const { agent } = await inquirer.prompt([
    {
      type: 'list',
      name: 'agent',
      message: 'Select agent:',
      choices: agents,
    },
  ]);

  console.log(chalk.green('✓ Switched to:'), agent, '\n');
  return agent;
}

module.exports = codeCommand;

const React = require('react');
const { render } = require('ink');
const chalk = require('chalk');
const inquirer = require('inquirer');
const ThauClient = require('../lib/client');
const SessionManager = require('../lib/session-manager');
const MultiConsole = require('../ui/MultiConsole');

/**
 * Multi-Console Code Command
 * Launches the multi-console UI with multiple agent sessions
 */
async function codeMultiCommand() {
  console.log(chalk.cyan.bold('\n💻 THAU Multi-Console Mode\n'));

  const client = new ThauClient();
  const sessionManager = new SessionManager();

  // Initialize .thau/ project config if not exists
  if (!client.projectInit.isInitialized()) {
    console.log(chalk.yellow('⚠️  Project not initialized'));
    const { init } = await inquirer.prompt([{
      type: 'confirm',
      name: 'init',
      message: 'Initialize THAU in this directory?',
      default: true
    }]);

    if (init) {
      await client.projectInit.initialize();
      console.log();
    }
  }

  // Check server health
  const healthy = await client.healthCheck();
  if (!healthy && client.currentModel === 'thau-api') {
    console.log(chalk.yellow('⚠️  Cannot connect to THAU server'));
    console.log(chalk.gray(`Server URL: ${client.serverUrl}`));

    const ollamaAvailable = await client.ollamaClient.isAvailable();
    if (ollamaAvailable) {
      console.log(chalk.green('✓ Ollama is available'));
      const { useOllama } = await inquirer.prompt([{
        type: 'confirm',
        name: 'useOllama',
        message: 'Use Ollama instead?',
        default: true
      }]);

      if (useOllama) {
        await client.switchModel('ollama');
        console.log(chalk.green('✓ Switched to Ollama'));
      } else {
        console.log(chalk.red('Cannot proceed without a model. Exiting.'));
        return;
      }
    } else {
      console.log(chalk.red('Neither THAU server nor Ollama are available. Exiting.'));
      return;
    }
  }

  const modelDisplay = client.currentModel === 'ollama' ?
    `Ollama (${client.currentOllamaModel})` :
    'THAU API';

  console.log(chalk.green(`Connected to: ${modelDisplay}`));
  console.log(chalk.gray('Launching multi-console UI...\n'));

  // Create initial sessions for common agents
  const defaultAgents = ['general', 'code_writer', 'planner'];
  defaultAgents.forEach((agent, idx) => {
    const sessionId = sessionManager.createSession(agent, client.currentModel);
    if (idx === 0) {
      sessionManager.switchSession(sessionId);
    }
  });

  // Render the Ink UI
  const { waitUntilExit } = render(
    React.createElement(MultiConsole, {
      sessionManager: sessionManager,
      thauClient: client,
      onExit: () => {
        console.log(chalk.yellow('\n👋 Goodbye!\n'));
      }
    })
  );

  // Wait for the app to exit
  await waitUntilExit();
}

module.exports = codeMultiCommand;

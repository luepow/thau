#!/usr/bin/env node

/**
 * THAU CODE - AI-Powered Coding Assistant CLI
 *
 * Like Claude Code, but powered by THAU AI
 * Node.js version
 */

const { program } = require('commander');
const chalk = require('chalk');
const boxen = require('boxen');
const pkg = require('../package.json');

// Import commands
const codeCommand = require('../src/commands/code');

// CLI Configuration
program
  .name('thau')
  .description('🧠 THAU CODE - AI-Powered Coding Assistant')
  .version(pkg.version);

// Show welcome screen if no command
if (process.argv.length === 2) {
  showWelcome();
  process.exit(0);
}

// Commands
program
  .command('code')
  .description('Interactive coding mode (like Claude Code)')
  .action(codeCommand);

// Parse arguments
program.parse(process.argv);

/**
 * Show welcome screen
 */
function showWelcome() {
  const welcome = `
${chalk.cyan.bold('🧠 THAU CODE')}

${chalk.dim('AI-Powered Coding Assistant')}

${chalk.dim('Commands:')}
  ${chalk.cyan('thau code')}              - Interactive coding mode (like Claude Code)
  ${chalk.cyan('thau --version')}         - Show version
  ${chalk.cyan('thau --help')}            - Show help

${chalk.dim('Features:')}
  • 11 Specialized Agents (code_writer, planner, debugger, etc.)
  • THAU API Server + Ollama support
  • Multi-turn conversations
  • MCP integration
  • Permission system

${chalk.dim('Start coding:')}
  ${chalk.white('thau code')}
`;

  console.log(
    boxen(welcome, {
      padding: 1,
      margin: 1,
      borderStyle: 'round',
      borderColor: 'cyan',
    })
  );
}

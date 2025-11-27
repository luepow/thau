#!/usr/bin/env node

const { program } = require('commander');
const chalk = require('chalk');
const boxen = require('boxen');

// Import command modules
const codeCommand = require('../src/commands/code');
const initCommand = require('../src/commands/init');
const createCommand = require('../src/commands/create');
const planCommand = require('../src/commands/plan');
const execCommand = require('../src/commands/exec');
const reviewCommand = require('../src/commands/review');
const refactorCommand = require('../src/commands/refactor');
const testCommand = require('../src/commands/test');
const chatCommand = require('../src/commands/chat');
const deployCommand = require('../src/commands/deploy');
const configCommand = require('../src/commands/config');

// Welcome screen
console.log(boxen(
  chalk.cyan.bold('THAU CLI') + '\n\n' +
  chalk.gray('AI-powered coding agent and toolkit\n') +
  chalk.gray('Similar to Claude Code, powered by THAU'),
  {
    padding: 1,
    borderStyle: 'round',
    borderColor: 'cyan',
    margin: 1
  }
));

// CLI configuration
program
  .name('thau')
  .description('THAU CLI - AI-powered coding agent')
  .version('1.0.0');

// Interactive coding mode
program
  .command('code')
  .description('Start interactive coding session')
  .action(codeCommand);

// Project initialization
program
  .command('init')
  .description('Initialize a new project from template')
  .option('-n, --name <name>', 'Project name')
  .option('-t, --template <template>', 'Template type (python, react, fastapi, etc.)')
  .action(initCommand);

// Create artifacts
program
  .command('create <type> [name]')
  .description('Create file/artifact (types: file, class, function, component)')
  .action(createCommand);

// Task planning
program
  .command('plan [task]')
  .description('Create a detailed plan for a task')
  .action(planCommand);

// Execute plan
program
  .command('exec [file]')
  .description('Execute a saved plan')
  .action(execCommand);

// Code review
program
  .command('review <files...>')
  .description('Review code for bugs and improvements')
  .action(reviewCommand);

// Refactor code
program
  .command('refactor <file>')
  .description('Refactor code with AI assistance')
  .action(refactorCommand);

// Generate tests
program
  .command('test <files...>')
  .description('Generate tests for code')
  .action(testCommand);

// Chat with file context
program
  .command('chat')
  .description('Chat about code')
  .option('-f, --file <file>', 'File to use as context')
  .action(chatCommand);

// Deploy
program
  .command('deploy')
  .description('Deploy project')
  .option('--env <environment>', 'Deployment environment')
  .action(deployCommand);

// Configuration
program
  .command('config')
  .description('View or update configuration')
  .option('--show', 'Show current configuration')
  .option('--server <url>', 'Set server URL')
  .option('--agent <name>', 'Set default agent')
  .action(configCommand);

// Parse command-line arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}

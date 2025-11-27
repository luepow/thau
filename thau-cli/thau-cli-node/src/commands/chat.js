const inquirer = require('inquirer');
const chalk = require('chalk');
const fs = require('fs-extra');
const marked = require('marked');
const { markedTerminal } = require('marked-terminal');
const ThauClient = require('../lib/client');

marked.use(markedTerminal());

async function chatCommand(options) {
  console.log(chalk.cyan.bold('\n💬 THAU Chat\n'));

  const client = new ThauClient();
  const conversationHistory = [];

  // Load file context if provided
  let fileContext = '';
  if (options.file) {
    if (!fs.existsSync(options.file)) {
      console.log(chalk.red(`File not found: ${options.file}`));
      return;
    }
    fileContext = fs.readFileSync(options.file, 'utf8');
    console.log(chalk.gray(`Loaded context from: ${options.file}\n`));
  }

  console.log(chalk.gray('Type your message or /exit to quit\n'));

  while (true) {
    const { message } = await inquirer.prompt([{
      type: 'input',
      name: 'message',
      message: chalk.cyan('You:'),
    }]);

    if (message === '/exit' || message === '/quit') {
      console.log(chalk.yellow('\n👋 Goodbye!\n'));
      break;
    }

    if (message === '/clear') {
      conversationHistory.length = 0;
      fileContext = '';
      console.log(chalk.green('✓ Conversation cleared'));
      continue;
    }

    if (message === '/save') {
      const filename = 'chat_history.json';
      fs.writeFileSync(filename, JSON.stringify(conversationHistory, null, 2));
      console.log(chalk.green(`✓ Conversation saved to: ${filename}`));
      continue;
    }

    if (message === '/help') {
      console.log(chalk.cyan(`
Available commands:
  /exit, /quit - Exit chat
  /clear       - Clear conversation history
  /save        - Save conversation to file
  /help        - Show this help message
      `));
      continue;
    }

    conversationHistory.push({
      role: 'user',
      content: message
    });

    try {
      // Build context
      const context = fileContext ? [
        { role: 'system', content: `File context:\n${fileContext}` },
        ...conversationHistory.slice(-5)
      ] : conversationHistory.slice(-5);

      const response = await client.sendTask(
        message || 'Analyze this code:',
        'general',
        context
      );

      const answer = response.result || response.description || 'No response';

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

module.exports = chatCommand;

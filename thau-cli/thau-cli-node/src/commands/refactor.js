module.exports = async function(file) {
  const chalk = require('chalk');
  const fs = require('fs-extra');
  const inquirer = require('inquirer');
  const ThauClient = require('../lib/client');

  console.log(chalk.cyan.bold('\n♻️ THAU Refactor\n'));

  if (!fs.existsSync(file)) {
    console.log(chalk.red(`File not found: ${file}`));
    return;
  }

  const code = fs.readFileSync(file, 'utf8');
  const client = new ThauClient();

  const { goals } = await inquirer.prompt([{
    type: 'input',
    name: 'goals',
    message: 'Refactoring goals:',
    default: 'Improve readability, remove code smells, follow SOLID principles'
  }]);

  console.log(chalk.cyan('\n🧠 THAU Refactorer working...\n'));

  const result = await client.sendTask(
    `Refactor this code. Goals: ${goals}\n\n\`\`\`\n${code}\n\`\`\``,
    'refactorer'
  );

  console.log(result.result || result.description || 'No result');

  const { save } = await inquirer.prompt([{
    type: 'confirm',
    name: 'save',
    message: '\nApply refactoring?',
    default: false
  }]);

  if (save) {
    fs.writeFileSync(file, result.result || code);
    console.log(chalk.green('✓ File updated'));
  }
};

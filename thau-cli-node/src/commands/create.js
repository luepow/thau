const inquirer = require('inquirer');
const chalk = require('chalk');
const fs = require('fs-extra');
const ThauClient = require('../lib/client');

async function createCommand(type, name) {
  console.log(chalk.cyan(`\nCreating ${type}: ${name}\n`));

  const client = new ThauClient();

  switch (type) {
    case 'file':
      await createFile(name);
      break;
    case 'class':
      await createClass(name, client);
      break;
    case 'function':
      await createFunction(name, client);
      break;
    case 'component':
      await createComponent(name, client);
      break;
    default:
      console.log(chalk.yellow(`Type ${type} not yet implemented`));
      await createFile(name);
  }
}

async function createFile(name) {
  const { content } = await inquirer.prompt([{
    type: 'input',
    name: 'content',
    message: 'File content (or press Enter for empty):',
  }]);

  fs.ensureDirSync(require('path').dirname(name));
  fs.writeFileSync(name, content || '');
  console.log(chalk.green('✓ Created:'), name);
}

async function createClass(name, client) {
  const { description, language } = await inquirer.prompt([
    { type: 'input', name: 'description', message: 'Describe the class:' },
    { type: 'list', name: 'language', message: 'Language:', choices: ['python', 'typescript', 'java'], default: 'typescript' }
  ]);

  console.log(chalk.cyan('\n🧠 THAU generating class...\n'));

  const response = await client.sendTask(
    `Create a ${language} class named ${name}. ${description}`,
    'code_writer'
  );

  if (response.error) {
    console.log(chalk.red('Error:'), response.error);
    return;
  }

  const code = response.result || response.description || '';
  console.log(code);

  const { save } = await inquirer.prompt([{
    type: 'confirm',
    name: 'save',
    message: 'Save to file?',
    default: true
  }]);

  if (save) {
    const ext = { python: '.py', typescript: '.ts', java: '.java' }[language];
    const filename = `${name}${ext}`;
    fs.writeFileSync(filename, code);
    console.log(chalk.green('✓ Saved to:'), filename);
  }
}

async function createFunction(name, client) {
  const { description } = await inquirer.prompt([
    { type: 'input', name: 'description', message: 'Describe the function:' }
  ]);

  console.log(chalk.cyan('\n🧠 THAU generating function...\n'));

  const response = await client.sendTask(
    `Create a TypeScript function named ${name}. ${description}`,
    'code_writer'
  );

  console.log(response.result || response.description || 'No result');
}

async function createComponent(name, client) {
  const { framework } = await inquirer.prompt([{
    type: 'list',
    name: 'framework',
    message: 'Framework:',
    choices: ['react', 'vue'],
    default: 'react'
  }]);

  console.log(chalk.cyan(`\n🧠 THAU generating ${framework} component...\n`));

  const response = await client.sendTask(
    `Create a ${framework} component named ${name}. Use TypeScript and modern best practices.`,
    'code_writer'
  );

  console.log(response.result || response.description || 'No result');
}

module.exports = createCommand;

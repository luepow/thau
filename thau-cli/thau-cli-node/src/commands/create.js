const inquirer = require('inquirer');
const chalk = require('chalk');
const fs = require('fs-extra');
const ora = require('ora');
const ThauClient = require('../lib/client');

async function createCommand(type, name) {
  console.log(chalk.cyan.bold('\n✨ THAU Creator\n'));

  if (!type) {
    const answer = await inquirer.prompt([{
      type: 'list',
      name: 'type',
      message: 'What do you want to create?',
      choices: ['file', 'class', 'function', 'component']
    }]);
    type = answer.type;
  }

  if (!name) {
    const answer = await inquirer.prompt([{
      type: 'input',
      name: 'name',
      message: `${type.charAt(0).toUpperCase() + type.slice(1)} name:`,
    }]);
    name = answer.name;
  }

  const client = new ThauClient();

  switch (type) {
    case 'file':
      await createFile(name, client);
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
      console.log(chalk.red(`Unknown type: ${type}`));
      console.log(chalk.gray('Supported types: file, class, function, component'));
  }
}

async function createFile(name, client) {
  const { description } = await inquirer.prompt([{
    type: 'input',
    name: 'description',
    message: 'Describe what this file should contain:',
  }]);

  const spinner = ora('Creating file...').start();

  try {
    const response = await client.sendTask(
      `Create a file named ${name}. ${description}`,
      'code_writer'
    );

    spinner.succeed('File created');

    const code = response.result || response.description || '';
    console.log(chalk.gray('\nGenerated code:\n'));
    console.log(code);

    const { save } = await inquirer.prompt([{
      type: 'confirm',
      name: 'save',
      message: '\nSave to file?',
      default: true
    }]);

    if (save) {
      fs.writeFileSync(name, code);
      console.log(chalk.green(`✓ Saved to: ${name}`));
    }

  } catch (error) {
    spinner.fail('Creation failed');
    console.log(chalk.red('Error:'), error.message);
  }
}

async function createClass(name, client) {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'description',
      message: 'Describe the class:',
    },
    {
      type: 'list',
      name: 'language',
      message: 'Programming language:',
      choices: ['python', 'typescript', 'java', 'javascript']
    }
  ]);

  const spinner = ora('Creating class...').start();

  try {
    const response = await client.sendTask(
      `Create a ${answers.language} class named ${name}. ${answers.description}`,
      'code_writer'
    );

    spinner.succeed('Class created');

    const code = response.result || response.description || '';
    console.log(chalk.gray('\nGenerated code:\n'));
    console.log(code);

    const { save } = await inquirer.prompt([{
      type: 'confirm',
      name: 'save',
      message: '\nSave to file?',
      default: true
    }]);

    if (save) {
      const extensions = {
        python: '.py',
        typescript: '.ts',
        java: '.java',
        javascript: '.js'
      };
      const filename = `${name}${extensions[answers.language]}`;
      fs.writeFileSync(filename, code);
      console.log(chalk.green(`✓ Saved to: ${filename}`));
    }

  } catch (error) {
    spinner.fail('Creation failed');
    console.log(chalk.red('Error:'), error.message);
  }
}

async function createFunction(name, client) {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'description',
      message: 'Describe what the function does:',
    },
    {
      type: 'list',
      name: 'language',
      message: 'Programming language:',
      choices: ['python', 'typescript', 'javascript', 'java']
    }
  ]);

  const spinner = ora('Creating function...').start();

  try {
    const response = await client.sendTask(
      `Create a ${answers.language} function named ${name}. ${answers.description}`,
      'code_writer'
    );

    spinner.succeed('Function created');

    const code = response.result || response.description || '';
    console.log(chalk.gray('\nGenerated code:\n'));
    console.log(code);

    const { save } = await inquirer.prompt([{
      type: 'confirm',
      name: 'save',
      message: '\nAppend to file?',
      default: true
    }]);

    if (save) {
      const { filename } = await inquirer.prompt([{
        type: 'input',
        name: 'filename',
        message: 'Filename:',
        default: `${name}.${answers.language === 'python' ? 'py' : 'ts'}`
      }]);

      if (fs.existsSync(filename)) {
        fs.appendFileSync(filename, '\n\n' + code);
      } else {
        fs.writeFileSync(filename, code);
      }
      console.log(chalk.green(`✓ Saved to: ${filename}`));
    }

  } catch (error) {
    spinner.fail('Creation failed');
    console.log(chalk.red('Error:'), error.message);
  }
}

async function createComponent(name, client) {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'description',
      message: 'Component specs:',
    },
    {
      type: 'list',
      name: 'framework',
      message: 'Framework:',
      choices: ['react', 'vue', 'svelte', 'angular']
    },
    {
      type: 'confirm',
      name: 'typescript',
      message: 'Use TypeScript?',
      default: true
    }
  ]);

  const spinner = ora('Creating component...').start();

  try {
    const response = await client.sendTask(
      `Create a ${answers.framework} component named ${name}${answers.typescript ? ' with TypeScript' : ''}. ${answers.description}`,
      'code_writer'
    );

    spinner.succeed('Component created');

    const code = response.result || response.description || '';
    console.log(chalk.gray('\nGenerated code:\n'));
    console.log(code);

    const { save } = await inquirer.prompt([{
      type: 'confirm',
      name: 'save',
      message: '\nSave to file?',
      default: true
    }]);

    if (save) {
      let extension;
      switch (answers.framework) {
        case 'react':
          extension = answers.typescript ? '.tsx' : '.jsx';
          break;
        case 'vue':
          extension = '.vue';
          break;
        case 'svelte':
          extension = '.svelte';
          break;
        case 'angular':
          extension = '.component.ts';
          break;
      }

      const filename = `${name}${extension}`;
      fs.writeFileSync(filename, code);
      console.log(chalk.green(`✓ Saved to: ${filename}`));
    }

  } catch (error) {
    spinner.fail('Creation failed');
    console.log(chalk.red('Error:'), error.message);
  }
}

module.exports = createCommand;

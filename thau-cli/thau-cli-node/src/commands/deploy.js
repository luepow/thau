const inquirer = require('inquirer');
const chalk = require('chalk');
const ora = require('ora');
const { exec } = require('child_process');
const util = require('util');
const ThauClient = require('../lib/client');

const execPromise = util.promisify(exec);

async function deployCommand(options) {
  console.log(chalk.cyan.bold('\n🚀 THAU Deploy\n'));

  const client = new ThauClient();

  let env = options.env;
  if (!env) {
    const answer = await inquirer.prompt([{
      type: 'list',
      name: 'environment',
      message: 'Select deployment environment:',
      choices: ['development', 'staging', 'production']
    }]);
    env = answer.environment;
  }

  console.log(chalk.cyan(`\nPreparing deployment to: ${env}\n`));

  // Pre-deployment checks
  const spinner = ora('Running pre-deployment checks...').start();

  try {
    const result = await client.sendTask(
      `Analyze the current project and provide deployment readiness checklist for ${env} environment`,
      'architect'
    );

    spinner.succeed('Pre-deployment analysis complete');
    console.log(result.result || result.description || 'No checklist generated');
    console.log();

    const { proceed } = await inquirer.prompt([{
      type: 'confirm',
      name: 'proceed',
      message: 'Proceed with deployment?',
      default: env !== 'production'
    }]);

    if (!proceed) {
      console.log(chalk.yellow('⊘ Deployment cancelled'));
      return;
    }

    // Build step
    const buildSpinner = ora('Building project...').start();

    try {
      const buildFlag = env === 'production' ? '--prod' : '';
      await execPromise(`npm run build ${buildFlag}`);
      buildSpinner.succeed('Build completed');
    } catch (error) {
      buildSpinner.fail('Build failed');
      console.log(chalk.red(error.message));
      return;
    }

    // Deploy step
    const deploySpinner = ora(`Deploying to ${env}...`).start();

    try {
      await execPromise(`npm run deploy:${env}`);
      deploySpinner.succeed(`Deployed to ${env}`);
      console.log(chalk.green.bold('\n✓ Deployment successful!\n'));
    } catch (error) {
      deploySpinner.fail('Deployment failed');
      console.log(chalk.red(error.message));
    }

  } catch (error) {
    spinner.fail('Pre-deployment check failed');
    console.log(chalk.red('Error:'), error.message);
  }
}

module.exports = deployCommand;

const inquirer = require('inquirer');
const chalk = require('chalk');
const ora = require('ora');
const fs = require('fs-extra');
const ThauClient = require('../lib/client');

async function execCommand(planFile) {
  console.log(chalk.cyan.bold('\n▶️  THAU Plan Executor\n'));

  if (!planFile) {
    const answer = await inquirer.prompt([{
      type: 'input',
      name: 'file',
      message: 'Plan file path:',
      default: 'plan.json'
    }]);
    planFile = answer.file;
  }

  if (!fs.existsSync(planFile)) {
    console.log(chalk.red(`Plan file not found: ${planFile}`));
    return;
  }

  const plan = JSON.parse(fs.readFileSync(planFile, 'utf8'));
  const client = new ThauClient();

  console.log(chalk.cyan.bold(`Executing plan: ${plan.task_description || 'N/A'}\n`));

  const steps = plan.steps || [];

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const stepDesc = typeof step === 'object' ? step.description : step;

    console.log(chalk.yellow(`\nStep ${i + 1}/${steps.length}: ${stepDesc}`));

    const { proceed } = await inquirer.prompt([{
      type: 'confirm',
      name: 'proceed',
      message: 'Execute this step?',
      default: true
    }]);

    if (!proceed) {
      console.log(chalk.yellow('⊘ Skipped'));
      continue;
    }

    const spinner = ora('Executing...').start();

    try {
      const result = await client.sendTask(
        `Execute this step: ${stepDesc}`,
        'code_writer'
      );

      spinner.succeed('Step completed');
      console.log(result.result || result.description || 'Done');

      const { saveOutput } = await inquirer.prompt([{
        type: 'confirm',
        name: 'saveOutput',
        message: 'Save output to file?',
        default: false
      }]);

      if (saveOutput) {
        const filename = `step_${i + 1}_output.txt`;
        fs.writeFileSync(filename, result.result || result.description || '');
        console.log(chalk.green('✓ Saved to:'), filename);
      }
    } catch (error) {
      spinner.fail('Step failed');
      console.log(chalk.red('Error:'), error.message);

      const { continueOnError } = await inquirer.prompt([{
        type: 'confirm',
        name: 'continueOnError',
        message: 'Continue with remaining steps?',
        default: false
      }]);

      if (!continueOnError) {
        console.log(chalk.yellow('\n⊘ Execution stopped'));
        return;
      }
    }
  }

  console.log(chalk.green.bold('\n✓ Plan execution completed!\n'));
}

module.exports = execCommand;

const inquirer = require('inquirer');
const chalk = require('chalk');
const Table = require('cli-table3');
const fs = require('fs-extra');
const ThauClient = require('../lib/client');

async function planCommand(taskDescription) {
  console.log(chalk.cyan.bold('\n📋 THAU Planner\n'));

  if (!taskDescription) {
    const answer = await inquirer.prompt([{
      type: 'input',
      name: 'task',
      message: 'Describe the task you want to plan:'
    }]);
    taskDescription = answer.task;
  }

  const client = new ThauClient();

  console.log(chalk.cyan(`\nCreating plan for: ${taskDescription}\n`));
  console.log(chalk.cyan('🧠 THAU Planner thinking...\n'));

  const plan = await client.createPlan(taskDescription);

  if (plan.error) {
    console.log(chalk.red('Error:'), plan.error);
    return;
  }

  displayPlan(plan);

  const { save } = await inquirer.prompt([{
    type: 'confirm',
    name: 'save',
    message: '\nSave plan to file?',
    default: true
  }]);

  if (save) {
    const safeName = taskDescription.replace(/[^a-zA-Z0-9 _-]/g, '').substring(0, 50).replace(/ /g, '_');
    const filename = `plan_${safeName}.json`;
    fs.writeFileSync(filename, JSON.stringify(plan, null, 2));
    console.log(chalk.green('✓ Plan saved to:'), filename);
  }
}

function displayPlan(plan) {
  console.log('='.repeat(70));
  console.log(chalk.cyan.bold('📋 DETAILED PLAN'));
  console.log('='.repeat(70) + '\n');

  console.log(chalk.bold('Task:'), plan.task_description || 'N/A');
  console.log();

  const steps = plan.steps || [];
  if (steps.length > 0) {
    const table = new Table({
      head: [chalk.cyan('#'), chalk.cyan('Description'), chalk.cyan('Est. Time')],
      colWidths: [5, 50, 15]
    });

    steps.forEach((step, i) => {
      const desc = typeof step === 'object' ? step.description : step;
      const time = typeof step === 'object' ? step.estimated_time : 'N/A';
      table.push([i + 1, desc, time]);
    });

    console.log(chalk.bold('Steps:\n'));
    console.log(table.toString());
    console.log();
  }

  if (plan.total_estimated_time) {
    console.log(chalk.bold('Total Estimated Time:'), chalk.yellow(plan.total_estimated_time));
    console.log();
  }
}

module.exports = planCommand;

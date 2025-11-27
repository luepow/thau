module.exports = async function(files) {
  const chalk = require('chalk');
  const fs = require('fs-extra');
  const ThauClient = require('../lib/client');

  console.log(chalk.cyan.bold('\n🔍 THAU Code Review\n'));

  if (!files || files.length === 0) {
    console.log(chalk.yellow('No files specified'));
    return;
  }

  const client = new ThauClient();

  for (const file of files) {
    if (!fs.existsSync(file)) {
      console.log(chalk.red(`File not found: ${file}`));
      continue;
    }

    const code = fs.readFileSync(file, 'utf8');
    console.log(chalk.cyan(`\nReviewing: ${file}\n`));

    const result = await client.sendTask(
      `Review this code for bugs, security issues, and improvements:\n\n\`\`\`\n${code}\n\`\`\``,
      'code_reviewer'
    );

    console.log(result.result || result.description || 'No review available');
    console.log();
  }
};

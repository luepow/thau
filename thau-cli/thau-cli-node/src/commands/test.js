module.exports = async function(files) {
  const chalk = require('chalk');
  const fs = require('fs-extra');
  const ThauClient = require('../lib/client');

  console.log(chalk.cyan.bold('\n🧪 THAU Test Generator\n'));

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
    console.log(chalk.cyan(`\nGenerating tests for: ${file}\n`));

    const result = await client.sendTask(
      `Generate comprehensive tests for this code:\n\n\`\`\`\n${code}\n\`\`\``,
      'test_writer'
    );

    console.log(result.result || result.description || 'No tests generated');
    console.log();
  }
};

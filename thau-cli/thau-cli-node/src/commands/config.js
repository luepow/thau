module.exports = async function(options) {
  const chalk = require('chalk');
  const boxen = require('boxen');
  const ThauClient = require('../lib/client');

  const client = new ThauClient();

  if (options.show) {
    const config = client.config;
    console.log(boxen(
      chalk.cyan('Server URL: ') + config.server_url + '\n' +
      chalk.cyan('Default Agent: ') + config.default_agent + '\n' +
      chalk.cyan('Theme: ') + config.theme + '\n' +
      chalk.cyan('Auto Save: ') + config.auto_save + '\n' +
      chalk.cyan('Auto Format: ') + config.auto_format,
      { padding: 1, borderStyle: 'round', borderColor: 'blue', title: '⚙️ THAU CODE Configuration' }
    ));
    return;
  }

  if (options.server) {
    client.config.server_url = options.server;
    console.log(chalk.green('✓ Server URL:'), options.server);
  }

  if (options.agent) {
    client.config.default_agent = options.agent;
    console.log(chalk.green('✓ Default agent:'), options.agent);
  }

  if (options.server || options.agent) {
    client.saveConfig(client.config);
  }
};

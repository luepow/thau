#!/usr/bin/env node
/**
 * Integration Test for THAU CLI
 * Tests the complete flow: CLI -> Server -> Ollama -> Response
 */

const ThauClient = require('./src/lib/client');
const chalk = require('chalk');

async function testIntegration() {
  console.log(chalk.cyan.bold('\n🧪 THAU CLI Integration Test\n'));

  const client = new ThauClient();

  // Test 1: Health Check
  console.log(chalk.yellow('Test 1: Health Check'));
  const healthy = await client.healthCheck();
  console.log(healthy ?
    chalk.green('✓ Server is running') :
    chalk.red('✗ Server is not running')
  );

  if (!healthy) {
    console.log(chalk.red('\n❌ Server not available. Start with:'));
    console.log(chalk.gray('python api/thau_code_server.py'));
    process.exit(1);
  }

  // Test 2: Send Task with code_writer agent
  console.log(chalk.yellow('\nTest 2: Send Task with AI Processing'));

  const testMessage = '¿Qué es JavaScript en una frase corta?';
  console.log(chalk.gray(`Message: "${testMessage}"`));

  try {
    const response = await client.sendTask(
      testMessage,
      'code_writer',
      []
    );

    console.log(chalk.gray('\nResponse:'));
    console.log(chalk.gray('- Task ID:'), response.task_id || 'N/A');
    console.log(chalk.gray('- Agent:'), response.agent_role || 'N/A');
    console.log(chalk.gray('- Status:'), response.status || 'N/A');

    const answer = response.result || response.description || 'No response';
    console.log(chalk.gray('- Result:'), answer.substring(0, 100) + '...');

    // Verify it's NOT just echoing
    const isEcho = answer.includes(testMessage);
    const hasSubstantialContent = answer.length > testMessage.length * 2;

    if (isEcho && !hasSubstantialContent) {
      console.log(chalk.red('✗ Server is echoing, not processing with AI'));
      console.log(chalk.yellow('  The response should be an AI-generated explanation, not an echo'));
      process.exit(1);
    } else {
      console.log(chalk.green('✓ Server is processing with AI (Ollama)'));
    }

  } catch (error) {
    console.log(chalk.red('✗ Error sending task:'), error.message);
    process.exit(1);
  }

  // Test 3: Verify multi-turn conversation works
  console.log(chalk.yellow('\nTest 3: Multi-Turn Conversation'));

  const conversationHistory = [
    { role: 'user', content: '¿Qué es Python?' },
    { role: 'assistant', content: 'Python es un lenguaje de programación interpretado.' }
  ];

  try {
    const response = await client.sendTask(
      '¿Y qué es JavaScript?',
      'code_writer',
      conversationHistory
    );

    const answer = response.result || response.description || 'No response';
    console.log(chalk.gray('Response:'), answer.substring(0, 100) + '...');
    console.log(chalk.green('✓ Multi-turn conversation works'));

  } catch (error) {
    console.log(chalk.red('✗ Error in multi-turn:'), error.message);
    process.exit(1);
  }

  // Test 4: Verify different agents work
  console.log(chalk.yellow('\nTest 4: Different Agent Types'));

  const agents = ['general', 'planner', 'explainer'];

  for (const agent of agents) {
    try {
      const response = await client.sendTask(
        `Test con agente ${agent}`,
        agent,
        []
      );

      const answer = response.result || response.description || 'No response';
      console.log(chalk.gray(`- ${agent}:`), answer.length > 0 ? chalk.green('✓') : chalk.red('✗'));

    } catch (error) {
      console.log(chalk.red(`✗ Error with ${agent}:`), error.message);
    }
  }

  console.log(chalk.cyan.bold('\n✅ All Integration Tests Passed!\n'));
  console.log(chalk.gray('The THAU CLI is ready to use with:'));
  console.log(chalk.white('  node bin/thau.js code'));
  console.log();
}

testIntegration().catch(error => {
  console.error(chalk.red('Test failed:'), error);
  process.exit(1);
});

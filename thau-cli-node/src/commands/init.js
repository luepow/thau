const inquirer = require('inquirer');
const chalk = require('chalk');
const fs = require('fs-extra');
const path = require('path');
const { execSync } = require('child_process');

const TEMPLATES = {
  'python': 'Python project',
  'fastapi': 'FastAPI web application',
  'flask': 'Flask web application',
  'react': 'React application with TypeScript',
  'nextjs': 'Next.js application',
  'vue': 'Vue.js application',
  'node': 'Node.js project',
  'express': 'Express.js API',
};

async function initCommand(options) {
  console.log(chalk.cyan.bold('\n🚀 THAU Project Initialization\n'));

  let { name, template } = options;

  if (!name) {
    const answers = await inquirer.prompt([{ type: 'input', name: 'projectName', message: 'Project name:' }]);
    name = answers.projectName;
  }

  if (!template) {
    const { selectedTemplate } = await inquirer.prompt([{
      type: 'list',
      name: 'selectedTemplate',
      message: 'Select template:',
      choices: Object.entries(TEMPLATES).map(([key, desc]) => ({ name: `${key} - ${desc}`, value: key }))
    }]);
    template = selectedTemplate;
  }

  console.log(chalk.cyan(`\n Creating ${template} project: ${name}\n`));

  switch (template) {
    case 'python':
      createPythonProject(name);
      break;
    case 'react':
      createReactProject(name);
      break;
    case 'node':
      createNodeProject(name);
      break;
    case 'express':
      createExpressProject(name);
      break;
    default:
      console.log(chalk.yellow(`Template ${template} coming soon...`));
      createNodeProject(name);
  }

  console.log(chalk.green(`\n✓ Project '${name}' created successfully!\n`));
  console.log(chalk.bold('Next steps:\n'));
  console.log(`  1. cd ${name}`);
  console.log(`  2. Start coding with THAU!\n`);
}

function createPythonProject(name) {
  fs.ensureDirSync(name);
  fs.ensureDirSync(path.join(name, 'src'));
  fs.ensureDirSync(path.join(name, 'tests'));

  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nPython project created with THAU\n`);
  fs.writeFileSync(path.join(name, 'requirements.txt'), '# Python dependencies\n');
  fs.writeFileSync(path.join(name, '.gitignore'), 'venv/\n__pycache__/\n*.pyc\n.env\n');
  fs.writeFileSync(path.join(name, 'src', 'main.py'), 'def main():\n    print("Hello from THAU!")\n\nif __name__ == "__main__":\n    main()\n');
}

function createNodeProject(name) {
  fs.ensureDirSync(name);
  fs.ensureDirSync(path.join(name, 'src'));

  const packageJson = {
    name,
    version: '1.0.0',
    description: 'Node.js project created with THAU',
    main: 'src/index.js',
    scripts: { start: 'node src/index.js' }
  };

  fs.writeFileSync(path.join(name, 'package.json'), JSON.stringify(packageJson, null, 2));
  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nNode.js project created with THAU\n`);
  fs.writeFileSync(path.join(name, '.gitignore'), 'node_modules/\n.env\n');
  fs.writeFileSync(path.join(name, 'src', 'index.js'), 'console.log("Hello from THAU!");\n');
}

function createReactProject(name) {
  console.log(chalk.cyan('Creating React app with create-react-app...'));
  execSync(`npx create-react-app ${name} --template typescript`, { stdio: 'inherit' });
}

function createExpressProject(name) {
  createNodeProject(name);

  const packageJsonPath = path.join(name, 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath));
  packageJson.dependencies = { express: '^4.18.0', cors: '^2.8.5' };
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));

  const serverCode = `const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ message: 'Hello from THAU!' });
});

app.listen(PORT, () => {
  console.log(\`Server running on port \${PORT}\`);
});
`;
  fs.writeFileSync(path.join(name, 'src', 'index.js'), serverCode);
}

module.exports = initCommand;

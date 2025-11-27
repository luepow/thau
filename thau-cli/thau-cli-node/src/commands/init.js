const inquirer = require('inquirer');
const chalk = require('chalk');
const fs = require('fs-extra');
const path = require('path');
const ora = require('ora');

const TEMPLATES = {
  'python': 'Python project with virtual environment',
  'fastapi': 'FastAPI web application',
  'flask': 'Flask web application',
  'react': 'React application with TypeScript',
  'nextjs': 'Next.js application',
  'vue': 'Vue.js application',
  'node': 'Node.js project',
  'express': 'Express.js API server',
};

async function initCommand(options) {
  console.log(chalk.cyan.bold('\n🚀 THAU Project Initializer\n'));

  let projectName = options.name;
  let template = options.template;

  if (!projectName) {
    const answer = await inquirer.prompt([{
      type: 'input',
      name: 'name',
      message: 'Project name:',
      default: 'my-app'
    }]);
    projectName = answer.name;
  }

  if (!template) {
    const answer = await inquirer.prompt([{
      type: 'list',
      name: 'template',
      message: 'Select template:',
      choices: Object.keys(TEMPLATES).map(key => ({
        name: `${key} - ${TEMPLATES[key]}`,
        value: key
      }))
    }]);
    template = answer.template;
  }

  if (!TEMPLATES[template]) {
    console.log(chalk.red(`Unknown template: ${template}`));
    console.log(chalk.gray('Available templates: ' + Object.keys(TEMPLATES).join(', ')));
    return;
  }

  const spinner = ora(`Creating ${template} project: ${projectName}`).start();

  try {
    // Create project directory
    fs.ensureDirSync(projectName);

    // Create based on template
    switch (template) {
      case 'python':
        createPythonProject(projectName);
        break;
      case 'fastapi':
        createFastAPIProject(projectName);
        break;
      case 'flask':
        createFlaskProject(projectName);
        break;
      case 'react':
        createReactProject(projectName);
        break;
      case 'nextjs':
        createNextJSProject(projectName);
        break;
      case 'vue':
        createVueProject(projectName);
        break;
      case 'node':
        createNodeProject(projectName);
        break;
      case 'express':
        createExpressProject(projectName);
        break;
    }

    spinner.succeed(`Project created: ${projectName}`);

    console.log(chalk.green('\n✓ Project initialized successfully!\n'));
    console.log(chalk.cyan('Next steps:'));
    console.log(chalk.gray(`  cd ${projectName}`));

    switch (template) {
      case 'python':
      case 'fastapi':
      case 'flask':
        console.log(chalk.gray('  python -m venv venv'));
        console.log(chalk.gray('  source venv/bin/activate'));
        console.log(chalk.gray('  pip install -r requirements.txt'));
        break;
      case 'react':
      case 'nextjs':
      case 'vue':
      case 'node':
      case 'express':
        console.log(chalk.gray('  npm install'));
        console.log(chalk.gray('  npm start'));
        break;
    }

    console.log();

  } catch (error) {
    spinner.fail('Project creation failed');
    console.log(chalk.red('Error:'), error.message);
  }
}

function createPythonProject(name) {
  fs.ensureDirSync(path.join(name, 'src'));
  fs.ensureDirSync(path.join(name, 'tests'));

  fs.writeFileSync(path.join(name, 'requirements.txt'), '# Add your dependencies here\n');
  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nPython project created with THAU CLI\n`);
  fs.writeFileSync(path.join(name, '.gitignore'), 'venv/\n__pycache__/\n*.pyc\n.env\n');
  fs.writeFileSync(path.join(name, 'src', '__init__.py'), '');
  fs.writeFileSync(path.join(name, 'src', 'main.py'), 'def main():\n    print("Hello from THAU!")\n\nif __name__ == "__main__":\n    main()\n');
}

function createFastAPIProject(name) {
  createPythonProject(name);

  const requirements = `fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0
`;

  const mainCode = `from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="${name}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello from THAU FastAPI!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
`;

  fs.writeFileSync(path.join(name, 'requirements.txt'), requirements);
  fs.writeFileSync(path.join(name, 'src', 'main.py'), mainCode);
}

function createFlaskProject(name) {
  createPythonProject(name);

  const requirements = `Flask==3.0.0
python-dotenv==1.0.0
`;

  const mainCode = `from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello from THAU Flask!"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
`;

  fs.writeFileSync(path.join(name, 'requirements.txt'), requirements);
  fs.writeFileSync(path.join(name, 'src', 'main.py'), mainCode);
}

function createNodeProject(name) {
  fs.ensureDirSync(path.join(name, 'src'));

  const packageJson = {
    name,
    version: '1.0.0',
    description: `${name} - Created with THAU CLI`,
    main: 'src/index.js',
    scripts: {
      start: 'node src/index.js',
      dev: 'nodemon src/index.js'
    },
    keywords: [],
    author: '',
    license: 'MIT',
    dependencies: {}
  };

  fs.writeJsonSync(path.join(name, 'package.json'), packageJson, { spaces: 2 });
  fs.writeFileSync(path.join(name, '.gitignore'), 'node_modules/\n.env\n');
  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nNode.js project created with THAU CLI\n`);
  fs.writeFileSync(path.join(name, 'src', 'index.js'), `console.log('Hello from THAU!');\n`);
}

function createExpressProject(name) {
  createNodeProject(name);

  const packageJson = fs.readJsonSync(path.join(name, 'package.json'));
  packageJson.dependencies = {
    express: '^4.18.2',
    cors: '^2.8.5',
    dotenv: '^16.3.1'
  };
  packageJson.scripts.start = 'node src/index.js';

  fs.writeJsonSync(path.join(name, 'package.json'), packageJson, { spaces: 2 });

  const serverCode = `const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ message: 'Hello from THAU Express!' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

app.listen(PORT, () => {
  console.log(\`Server running on port \${PORT}\`);
});
`;

  fs.writeFileSync(path.join(name, 'src', 'index.js'), serverCode);
  fs.writeFileSync(path.join(name, '.env'), 'PORT=3000\n');
}

function createReactProject(name) {
  fs.ensureDirSync(path.join(name, 'src'));
  fs.ensureDirSync(path.join(name, 'public'));

  const packageJson = {
    name,
    version: '1.0.0',
    description: `${name} - Created with THAU CLI`,
    scripts: {
      start: 'react-scripts start',
      build: 'react-scripts build',
      test: 'react-scripts test'
    },
    dependencies: {
      react: '^18.2.0',
      'react-dom': '^18.2.0',
      'react-scripts': '^5.0.1',
      typescript: '^5.0.0',
      '@types/react': '^18.2.0',
      '@types/react-dom': '^18.2.0'
    },
    eslintConfig: {
      extends: ['react-app']
    },
    browserslist: {
      production: ['>0.2%', 'not dead'],
      development: ['last 1 chrome version']
    }
  };

  fs.writeJsonSync(path.join(name, 'package.json'), packageJson, { spaces: 2 });
  fs.writeFileSync(path.join(name, '.gitignore'), 'node_modules/\nbuild/\n.env\n');
  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nReact project created with THAU CLI\n`);
  fs.writeFileSync(path.join(name, 'public', 'index.html'), `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${name}</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
`);

  fs.writeFileSync(path.join(name, 'src', 'index.tsx'), `import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
`);

  fs.writeFileSync(path.join(name, 'src', 'App.tsx'), `import React from 'react';

function App() {
  return (
    <div>
      <h1>Hello from THAU React!</h1>
    </div>
  );
}

export default App;
`);

  fs.writeFileSync(path.join(name, 'tsconfig.json'), `{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
`);
}

function createNextJSProject(name) {
  fs.ensureDirSync(path.join(name, 'app'));
  fs.ensureDirSync(path.join(name, 'public'));

  const packageJson = {
    name,
    version: '1.0.0',
    description: `${name} - Created with THAU CLI`,
    scripts: {
      dev: 'next dev',
      build: 'next build',
      start: 'next start'
    },
    dependencies: {
      next: '^14.0.0',
      react: '^18.2.0',
      'react-dom': '^18.2.0',
      typescript: '^5.0.0',
      '@types/react': '^18.2.0',
      '@types/react-dom': '^18.2.0'
    }
  };

  fs.writeJsonSync(path.join(name, 'package.json'), packageJson, { spaces: 2 });
  fs.writeFileSync(path.join(name, '.gitignore'), 'node_modules/\n.next/\nout/\n.env\n');
  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nNext.js project created with THAU CLI\n`);

  fs.writeFileSync(path.join(name, 'app', 'page.tsx'), `export default function Home() {
  return (
    <main>
      <h1>Hello from THAU Next.js!</h1>
    </main>
  );
}
`);

  fs.writeFileSync(path.join(name, 'app', 'layout.tsx'), `export const metadata = {
  title: '${name}',
  description: 'Created with THAU CLI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
`);

  fs.writeFileSync(path.join(name, 'tsconfig.json'), `{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
`);
}

function createVueProject(name) {
  fs.ensureDirSync(path.join(name, 'src'));
  fs.ensureDirSync(path.join(name, 'public'));

  const packageJson = {
    name,
    version: '1.0.0',
    description: `${name} - Created with THAU CLI`,
    scripts: {
      dev: 'vite',
      build: 'vite build',
      preview: 'vite preview'
    },
    dependencies: {
      vue: '^3.3.0'
    },
    devDependencies: {
      '@vitejs/plugin-vue': '^4.4.0',
      vite: '^5.0.0',
      typescript: '^5.0.0'
    }
  };

  fs.writeJsonSync(path.join(name, 'package.json'), packageJson, { spaces: 2 });
  fs.writeFileSync(path.join(name, '.gitignore'), 'node_modules/\ndist/\n.env\n');
  fs.writeFileSync(path.join(name, 'README.md'), `# ${name}\n\nVue.js project created with THAU CLI\n`);

  fs.writeFileSync(path.join(name, 'public', 'index.html'), `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${name}</title>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.ts"></script>
</body>
</html>
`);

  fs.writeFileSync(path.join(name, 'src', 'main.ts'), `import { createApp } from 'vue';
import App from './App.vue';

createApp(App).mount('#app');
`);

  fs.writeFileSync(path.join(name, 'src', 'App.vue'), `<template>
  <div>
    <h1>Hello from THAU Vue!</h1>
  </div>
</template>

<script setup lang="ts">
</script>
`);

  fs.writeFileSync(path.join(name, 'vite.config.ts'), `import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
});
`);
}

module.exports = initCommand;

/**
 * THAU Code Desktop - Monaco Code Editor Component
 */

import React, { useRef } from 'react';
import Editor, { Monaco } from '@monaco-editor/react';
import * as monaco from 'monaco-editor';

interface CodeEditorProps {
  value?: string;
  language?: string;
  theme?: 'vs-dark' | 'light';
  onChange?: (value: string | undefined) => void;
  readOnly?: boolean;
  height?: string;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  value = '',
  language = 'python',
  theme = 'vs-dark',
  onChange,
  readOnly = false,
  height = '600px',
}) => {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);

  const handleEditorDidMount = (
    editor: monaco.editor.IStandaloneCodeEditor,
    monaco: Monaco
  ) => {
    editorRef.current = editor;

    // Configure editor options
    editor.updateOptions({
      fontSize: 14,
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      wordWrap: 'on',
      readOnly,
      automaticLayout: true,
    });

    // Register custom themes if needed
    monaco.editor.defineTheme('thau-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [],
      colors: {
        'editor.background': '#1e1e1e',
      },
    });
  };

  return (
    <div style={{ border: '1px solid #3e3e3e', borderRadius: '4px', overflow: 'hidden' }}>
      <Editor
        height={height}
        language={language}
        theme={theme === 'vs-dark' ? 'thau-dark' : 'light'}
        value={value}
        onChange={onChange}
        onMount={handleEditorDidMount}
        options={{
          readOnly,
          fontSize: 14,
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          automaticLayout: true,
        }}
      />
    </div>
  );
};

export default CodeEditor;

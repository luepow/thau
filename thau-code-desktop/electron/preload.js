/**
 * THAU Code Desktop - Preload Script
 *
 * Exposes safe APIs from Electron to the renderer process
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Get THAU backend URL
  getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),

  // Open external link
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  // File operations
  readFile: (filePath) => ipcRenderer.invoke('read-file', filePath),
  writeFile: (filePath, content) => ipcRenderer.invoke('write-file', filePath, content),

  // Platform info
  platform: process.platform,

  // Version info
  versions: {
    node: process.versions.node,
    chrome: process.versions.chrome,
    electron: process.versions.electron,
  }
});

console.log('🔒 THAU Code preload script loaded');
console.log(`   Platform: ${process.platform}`);

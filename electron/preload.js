const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("noiseApi", {
  pickModel: () => ipcRenderer.invoke("dialog:pick-model"),
  pickInput: () => ipcRenderer.invoke("dialog:pick-input"),
  pickOutput: (inputPath, outputPath) => ipcRenderer.invoke("dialog:pick-output", inputPath, outputPath),
  buildDefaultOutput: (inputPath, mode) => ipcRenderer.invoke("path:default-output", inputPath, mode),
  saveCopy: (sourcePath, targetPath) => ipcRenderer.invoke("file:copy", sourcePath, targetPath),
  listModels: (mode) => ipcRenderer.invoke("backend:list-models", mode),
  getDefaultModel: (mode) => ipcRenderer.invoke("backend:get-default-model", mode),
  loadPreview: (inputPath) => ipcRenderer.invoke("backend:load-preview", inputPath),
  denoise: (payload) => ipcRenderer.invoke("backend:denoise", payload),
  cancelDenoise: () => ipcRenderer.invoke("backend:cancel-denoise"),
  onDenoiseProgress: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("backend:denoise-progress", listener);
    return () => ipcRenderer.removeListener("backend:denoise-progress", listener);
  }
});

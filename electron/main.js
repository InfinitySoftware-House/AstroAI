const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

const ROOT_DIR = app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "..");
const BACKEND_DIR = app.isPackaged ? path.join(process.resourcesPath, "backend") : ROOT_DIR;
const BRIDGE_SCRIPT = path.join(BACKEND_DIR, "desktop_bridge.py");
const activeDenoiseJobs = new Map();
let denoiseWorker = null;
let denoiseWorkerSeq = 0;

const SUPPORTED_INPUTS = [
  { name: "Supported images", extensions: ["fits", "fit", "png", "jpg", "jpeg", "tif", "tiff"] },
  { name: "All files", extensions: ["*"] }
];

const SUPPORTED_OUTPUTS = [
  { name: "FITS", extensions: ["fits", "fit"] },
  { name: "TIFF", extensions: ["tif", "tiff"] },
  { name: "PNG", extensions: ["png"] },
  { name: "JPEG", extensions: ["jpg", "jpeg"] },
  { name: "All files", extensions: ["*"] }
];

function resolvePythonExecutable() {
  const candidates = [
    path.join(ROOT_DIR, ".venv311", "Scripts", "python.exe"),
    path.join(ROOT_DIR, ".venv", "Scripts", "python.exe"),
    "python"
  ];
  return candidates.find((candidate) => candidate === "python" || fs.existsSync(candidate));
}

function buildDefaultOutput(inputPath, mode = "denoise") {
  if (!inputPath) {
    const outputDir = app.isPackaged ? app.getPath("documents") : ROOT_DIR;
    if (mode === "gradient") {
      return path.join(outputDir, "gradient_removed_output.tif");
    }
    if (mode === "star") {
      return path.join(outputDir, "stars_reduced_output.tif");
    }
    if (mode === "sharpen") {
      return path.join(outputDir, "sharpened_output.tif");
    }
    return path.join(outputDir, "denoised_output.tif");
  }
  const parsed = path.parse(inputPath);
  const suffix =
    mode === "gradient"
      ? "_gradient_removed"
      : mode === "star"
      ? "_stars_reduced"
      : mode === "sharpen"
      ? "_sharpened"
      : "_denoised";
  return path.join(parsed.dir, `${parsed.name}${suffix}${parsed.ext || ".tif"}`);
}

function runBridge(args, options = {}) {
  const pythonExecutable = resolvePythonExecutable();
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable, [BRIDGE_SCRIPT, ...args], {
      cwd: ROOT_DIR,
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";
    let stderrBuffer = "";

    const PROGRESS_MARK = "__PROGRESS__";

    const flushStderrLine = (line) => {
      if (!line) {
        return;
      }
      const markIndex = line.indexOf(PROGRESS_MARK);
      if (markIndex !== -1) {
        const jsonPart = line.slice(markIndex + PROGRESS_MARK.length);
        try {
          options.onProgress?.(JSON.parse(jsonPart));
          return;
        } catch (error) {
          stderr += `${line}\n`;
          return;
        }
      }
      stderr += `${line}\n`;
    };

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderrBuffer += chunk.toString();
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() || "";
      for (const line of lines) {
        flushStderrLine(line.trim());
      }
    });

    options.onSpawn?.(child);

    child.on("error", (error) => reject(error));
    child.on("close", (code) => {
      flushStderrLine(stderrBuffer.trim());
      if (options.isCancelled?.()) {
        reject(new Error("Denoising canceled."));
        return;
      }
      if (code !== 0) {
        reject(new Error(stderr.trim() || stdout.trim() || `Python bridge exited with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Invalid bridge response: ${stdout || error.message}`));
      }
    });
  });
}

function createDenoiseWorker() {
  const pythonExecutable = resolvePythonExecutable();
  const child = spawn(pythonExecutable, [BRIDGE_SCRIPT, "serve-denoise"], {
    cwd: ROOT_DIR,
    windowsHide: true
  });

  const workerState = {
    child,
    stdoutBuffer: "",
    stderrBuffer: "",
    pending: new Map()
  };

  const PROGRESS_MARK = "__PROGRESS__";

  const handleWorkerStderrLine = (line) => {
    if (!line) {
      return;
    }
    const markIndex = line.indexOf(PROGRESS_MARK);
    if (markIndex === -1) {
      for (const pending of workerState.pending.values()) {
        pending.lastError = pending.lastError ? `${pending.lastError}\n${line}` : line;
      }
      return;
    }

    const jsonPart = line.slice(markIndex + PROGRESS_MARK.length);
    try {
      const payload = JSON.parse(jsonPart);
      const requestId = String(payload.request_id || "");
      if (!requestId) {
        return;
      }
      const pending = workerState.pending.get(requestId);
      pending?.onProgress?.(payload);
    } catch (_error) {
      for (const pending of workerState.pending.values()) {
        pending.lastError = pending.lastError ? `${pending.lastError}\n${line}` : line;
      }
    }
  };

  child.stdout.on("data", (chunk) => {
    workerState.stdoutBuffer += chunk.toString();
    const lines = workerState.stdoutBuffer.split(/\r?\n/);
    workerState.stdoutBuffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        const payload = JSON.parse(trimmed);
        const requestId = String(payload.id || "");
        const pending = workerState.pending.get(requestId);
        if (!pending) {
          continue;
        }
        workerState.pending.delete(requestId);
        if (payload.ok) {
          pending.resolve(payload.result);
        } else {
          pending.reject(new Error(payload.error || pending.lastError || "Denoise worker request failed."));
        }
      } catch (_error) {
        for (const pending of workerState.pending.values()) {
          pending.lastError = pending.lastError ? `${pending.lastError}\n${trimmed}` : trimmed;
        }
      }
    }
  });

  child.stderr.on("data", (chunk) => {
    workerState.stderrBuffer += chunk.toString();
    const lines = workerState.stderrBuffer.split(/\r?\n/);
    workerState.stderrBuffer = lines.pop() || "";
    for (const line of lines) {
      handleWorkerStderrLine(line.trim());
    }
  });

  child.on("error", (error) => {
    for (const pending of workerState.pending.values()) {
      pending.reject(error);
    }
    workerState.pending.clear();
    if (denoiseWorker === workerState) {
      denoiseWorker = null;
    }
  });

  child.on("close", (code) => {
    const trailing = workerState.stderrBuffer.trim();
    if (trailing) {
      handleWorkerStderrLine(trailing);
    }
    for (const pending of workerState.pending.values()) {
      pending.reject(new Error(pending.lastError || `Denoise worker exited with code ${code}`));
    }
    workerState.pending.clear();
    if (denoiseWorker === workerState) {
      denoiseWorker = null;
    }
  });

  return workerState;
}

function ensureDenoiseWorker() {
  if (!denoiseWorker || !denoiseWorker.child || denoiseWorker.child.killed) {
    denoiseWorker = createDenoiseWorker();
  }
  return denoiseWorker;
}

function stopDenoiseWorker() {
  if (!denoiseWorker || !denoiseWorker.child || denoiseWorker.child.killed) {
    denoiseWorker = null;
    return;
  }
  denoiseWorker.child.kill();
  denoiseWorker = null;
}

function runDenoiseViaWorker(payload, options = {}) {
  const worker = ensureDenoiseWorker();
  const requestId = `req-${++denoiseWorkerSeq}`;
  return new Promise((resolve, reject) => {
    worker.pending.set(requestId, {
      resolve,
      reject,
      onProgress: options.onProgress,
      lastError: ""
    });

      worker.child.stdin.write(`${JSON.stringify({
      id: requestId,
      mode: payload.mode || "denoise",
      model_path: payload.modelPath,
      input: payload.inputPath,
      output: payload.outputPath,
      patch_size: payload.patchSize,
      stride: payload.stride,
      tta: payload.tta,
      batch_size: payload.batchSize,
      amp: Boolean(payload.amp),
      strength: payload.strength,
      detail_preservation: payload.detailPreservation,
      background_threshold: payload.backgroundThreshold,
      background_strength: payload.backgroundStrength,
      subject_detail_preservation: payload.subjectDetailPreservation,
      background_detail_preservation: payload.backgroundDetailPreservation,
      gradient_blur_sigma: payload.gradientBlurSigma !== undefined ? payload.gradientBlurSigma : 3.0,
      device: payload.device || ""
    })}\n`);
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1220,
    minHeight: 820,
    show: false,
    backgroundColor: "#0b1320",
    title: "DeepSkyDenoiser",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.once("ready-to-show", () => {
    win.maximize();
    win.show();
  });

  win.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(() => {
  ipcMain.handle("dialog:pick-model", async () => {
    const result = await dialog.showOpenDialog({
      properties: ["openFile"],
      filters: [{ name: "PyTorch checkpoint", extensions: ["pt", "pth"] }, { name: "All files", extensions: ["*"] }]
    });
    return result.canceled ? "" : result.filePaths[0];
  });

  ipcMain.handle("dialog:pick-input", async () => {
    const result = await dialog.showOpenDialog({
      properties: ["openFile"],
      filters: SUPPORTED_INPUTS
    });
    return result.canceled ? "" : result.filePaths[0];
  });

  ipcMain.handle("dialog:pick-output", async (_event, currentInput, currentOutput) => {
    const result = await dialog.showSaveDialog({
      defaultPath: currentOutput || buildDefaultOutput(currentInput),
      filters: SUPPORTED_OUTPUTS
    });
    return result.canceled ? "" : result.filePath;
  });

  ipcMain.handle("path:default-output", async (_event, inputPath, mode) => buildDefaultOutput(inputPath, mode));
  ipcMain.handle("file:copy", async (_event, sourcePath, targetPath) => {
    await fs.promises.copyFile(sourcePath, targetPath);
    return { outputPath: targetPath };
  });
  ipcMain.handle("backend:list-models", async (_event, mode) => runBridge(["list-models", "--mode", mode || "denoise"]));
  ipcMain.handle("backend:get-default-model", async (_event, mode) => runBridge(["get-default-model", "--mode", mode || "denoise"]));
  ipcMain.handle("backend:load-preview", async (_event, inputPath) => runBridge(["preview", "--input", inputPath]));
  ipcMain.handle("backend:denoise", async (_event, payload) => {
    const jobKey = _event.sender.id;
    if (activeDenoiseJobs.has(jobKey)) {
      throw new Error("A denoising job is already running.");
    }
    const jobState = { cancelled: false };
    activeDenoiseJobs.set(jobKey, jobState);
    let modelPath = payload.modelPath;
    if (!modelPath) {
      const defaultModel = await runBridge(["get-default-model", "--mode", payload.mode || "denoise"]);
      modelPath = defaultModel.model_path;
    }
    if (!modelPath) {
      throw new Error("No preloaded model was found.");
    }
    payload.modelPath = modelPath;
    const args = [
      "denoise",
      "--model-path", modelPath,
      "--input", payload.inputPath,
      "--output", payload.outputPath,
      "--patch-size", String(payload.patchSize),
      "--stride", String(payload.stride),
      "--tta", String(payload.tta),
      "--batch-size", String(payload.batchSize),
      "--strength", String(payload.strength),
      "--detail-preservation", String(payload.detailPreservation),
      "--background-threshold", String(payload.backgroundThreshold),
      "--background-strength", String(payload.backgroundStrength),
      "--subject-detail-preservation", String(payload.subjectDetailPreservation),
      "--background-detail-preservation", String(payload.backgroundDetailPreservation),
      "--device", payload.device || ""
    ];
    if (payload.amp) {
      args.push("--amp");
    }
    try {
      return await runDenoiseViaWorker(payload, {
        onProgress(progressPayload) {
          _event.sender.send("backend:denoise-progress", progressPayload);
        }
      });
    } finally {
      activeDenoiseJobs.delete(jobKey);
    }
  });
  ipcMain.handle("backend:cancel-denoise", async (_event) => {
    const jobState = activeDenoiseJobs.get(_event.sender.id);
    if (!jobState) {
      return { canceled: false };
    }
    jobState.cancelled = true;
    stopDenoiseWorker();
    return { canceled: true };
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopDenoiseWorker();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

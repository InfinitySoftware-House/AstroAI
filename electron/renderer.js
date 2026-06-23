const elements = {
  modeSelect: document.getElementById("modeSelect"),
  appSubtitle: document.getElementById("appSubtitle"),
  resultLabel: document.getElementById("resultLabel"),
  outputTooltipText: document.getElementById("outputTooltipText"),
  strengthTooltipText: document.getElementById("strengthTooltipText"),
  inputPath: document.getElementById("inputPath"),
  outputPath: document.getElementById("outputPath"),
  patchSize: document.getElementById("patchSize"),
  stride: document.getElementById("stride"),
  tta: document.getElementById("tta"),
  batchSize: document.getElementById("batchSize"),
  device: document.getElementById("device"),
  strength: document.getElementById("strength"),
  detailPreservation: document.getElementById("detailPreservation"),
  backgroundThreshold: document.getElementById("backgroundThreshold"),
  backgroundStrength: document.getElementById("backgroundStrength"),
  subjectDetailPreservation: document.getElementById("subjectDetailPreservation"),
  backgroundDetailPreservation: document.getElementById("backgroundDetailPreservation"),
  gradientBlurSigma: document.getElementById("gradientBlurSigma"),
  gradientBlurSigmaValue: document.getElementById("gradientBlurSigmaValue"),
  gradientOnlySettings: document.getElementById("gradientOnlySettings"),
  amp: document.getElementById("amp"),
  pickInput: document.getElementById("pickInput"),
  pickOutput: document.getElementById("pickOutput"),
  quickPresetButtons: Array.from(document.querySelectorAll(".quick-preset-btn")),
  runDenoise: document.getElementById("runDenoise"),
  cancelDenoise: document.getElementById("cancelDenoise"),
  saveResult: document.getElementById("saveResult"),
  statusText: document.getElementById("statusText"),
  progressMeta: document.getElementById("progressMeta"),
  progressLabel: document.getElementById("progressLabel"),
  progressPercent: document.getElementById("progressPercent"),
  progressBar: document.getElementById("progressBar"),
  modelSelect: document.getElementById("modelSelect"),
  variantStandard: document.getElementById("variantStandard"),
  variantLite: document.getElementById("variantLite"),
  backendName: document.getElementById("backendName"),
  backendDevice: document.getElementById("backendDevice"),
  originalPreview: document.getElementById("originalPreview"),
  denoisedPreview: document.getElementById("denoisedPreview"),
  originalEmpty: document.getElementById("originalEmpty"),
  compareCanvas: document.getElementById("compareCanvas"),
  compareClip: document.getElementById("compareClip"),
  compareDivider: document.getElementById("compareDivider")
};

const state = {
  busy: false,
  hasResult: false,
  lastSavedOutputPath: "",
  modelPath: "",
  mode: "denoise",
  modelVariant: "standard",
  allModels: [],
  sliderPct: 50
};

const QUICK_PRESETS = {
  fast: {
    label: "Fast",
    status: "Fast preset applied: fewer passes and lighter overlap for much quicker previews and trial runs.",
    values: {
      patchSize: 128,
      stride: 64,
      tta: 1,
      batchSize: 32,
      strength: 0.95,
      detailPreservation: 0.24,
      backgroundThreshold: 0.12,
      backgroundStrength: 1.15,
      subjectDetailPreservation: 0.24,
      backgroundDetailPreservation: 0.05
    }
  },
  balanced: {
    label: "Balanced",
    status: "Balanced preset applied: moderate overlap with a small TTA boost for a better speed-quality compromise.",
    values: {
      patchSize: 128,
      stride: 64,
      tta: 2,
      batchSize: 32,
      strength: 0.95,
      detailPreservation: 0.3,
      backgroundThreshold: 0.12,
      backgroundStrength: 1.2,
      subjectDetailPreservation: 0.32,
      backgroundDetailPreservation: 0.05
    }
  },
  galaxy: {
    label: "Galaxy",
    status: "Galaxy preset applied: stronger dark-background cleanup with gentle subject detail protection at balanced speed.",
    values: {
      patchSize: 128,
      stride: 64,
      tta: 2,
      batchSize: 32,
      strength: 0.95,
      detailPreservation: 0.35,
      backgroundThreshold: 0.12,
      backgroundStrength: 1.25,
      subjectDetailPreservation: 0.35,
      backgroundDetailPreservation: 0.04
    }
  },
  "globular-cluster": {
    label: "Globular Cluster",
    status: "Globular Cluster preset applied: protects dense star cores and keeps point sources tighter without the heaviest runtime cost.",
    values: {
      patchSize: 128,
      stride: 64,
      tta: 2,
      batchSize: 32,
      strength: 0.88,
      detailPreservation: 0.48,
      backgroundThreshold: 0.10,
      backgroundStrength: 1.05,
      subjectDetailPreservation: 0.56,
      backgroundDetailPreservation: 0.08
    }
  },
  "deep-field": {
    label: "Deep Field",
    status: "Deep Field preset applied: pushes background cleanup harder while keeping faint structures from disappearing, with runtime kept in check.",
    values: {
      patchSize: 128,
      stride: 64,
      tta: 2,
      batchSize: 24,
      strength: 1.02,
      detailPreservation: 0.30,
      backgroundThreshold: 0.16,
      backgroundStrength: 1.42,
      subjectDetailPreservation: 0.34,
      backgroundDetailPreservation: 0.02
    }
  },
  nebulae: {
    label: "Nebulae",
    status: "Nebulae preset applied: softer cleanup that preserves broad clouds and filament structure at balanced speed.",
    values: {
      patchSize: 128,
      stride: 64,
      tta: 2,
      batchSize: 32,
      strength: 0.90,
      detailPreservation: 0.52,
      backgroundThreshold: 0.08,
      backgroundStrength: 1.10,
      subjectDetailPreservation: 0.62,
      backgroundDetailPreservation: 0.10
    }
  }
};

function setStatus(message) {
  elements.statusText.textContent = message;
}

function getModeCopy(mode = state.mode) {
  if (mode === "gradient") {
    return {
      noun: "gradient removal",
      verb: "gradient correction",
      resultLabel: "Corrected",
      subtitle: "Load an image, remove background gradients, and save the corrected result.",
      outputTooltip: "Where the gradient-corrected image will be saved. A default name is created automatically from the input file.",
      strengthTooltip: "Controls how strongly the result moves toward the gradient-corrected version. Lower keeps more of the original background trend, higher removes more.",
      readyStatus: "Choose a gradient-removal model and an image, then run correction."
    };
  }
  if (mode === "star") {
    return {
      noun: "star reduction",
      verb: "star reduction",
      resultLabel: "Stars Reduced",
      subtitle: "Load an image, reduce bloated stars, and save the corrected result.",
      outputTooltip: "Where the star-reduced image will be saved. A default name is created automatically from the input file.",
      strengthTooltip: "Controls how strongly the result moves toward the star-reduced version. Lower keeps more of the original star size, higher reduces stars more.",
      readyStatus: "Choose a star-reducer model and an image, then run star reduction."
    };
  }
  if (mode === "sharpen") {
    return {
      noun: "sharpening",
      verb: "sharpening",
      resultLabel: "Sharpened",
      subtitle: "Load an image, recover fine detail and tighten stars, and save the result.",
      outputTooltip: "Where the sharpened image will be saved. A default name is created automatically from the input file.",
      strengthTooltip: "Controls how strongly the result moves toward the sharpened version. Lower keeps the image softer, higher recovers more detail.",
      readyStatus: "Choose a sharpen model and an image, then run sharpening."
    };
  }
  return {
    noun: "denoising",
    verb: "denoising",
    resultLabel: "Denoised",
    subtitle: "Load an image, keep the settings simple, and run.",
    outputTooltip: "Where the denoised image will be saved. A default name is created automatically from the input file.",
    strengthTooltip: "Controls how strongly the result moves toward the denoised version. Lower keeps more original noise, higher smooths more.",
    readyStatus: "Choose a model and an image, then run denoising."
  };
}

function inferBackendFromModelPath(modelPath) {
  const lower = String(modelPath || "").trim().toLowerCase();
  if (lower.endsWith(".onnx")) {
    return "ONNX Runtime";
  }
  if (lower.endsWith(".pt") || lower.endsWith(".pth")) {
    return "PyTorch";
  }
  return "Unknown";
}

function inferDeviceLabel(deviceValue, backend) {
  const value = String(deviceValue || "").trim().toLowerCase();
  if (!value) {
    return backend === "ONNX Runtime" ? "Auto (CPU/CUDA if available)" : "Auto (prefers CUDA)";
  }
  if (value.startsWith("cuda")) {
    return `CUDA (${value})`;
  }
  if (value === "cpu") {
    return "CPU";
  }
  return value;
}

function updateBackendBadge() {
  const backend = inferBackendFromModelPath(state.modelPath);
  const deviceLabel = inferDeviceLabel(elements.device.value, backend);
  elements.backendName.textContent = `${backend}`;
  elements.backendDevice.textContent = `${deviceLabel}`;
}

function setBusy(busy) {
  state.busy = busy;
  for (const node of [
    elements.inputPath,
    elements.outputPath,
    elements.patchSize,
    elements.stride,
    elements.tta,
    elements.batchSize,
    elements.device,
    elements.strength,
    elements.detailPreservation,
    elements.backgroundThreshold,
    elements.backgroundStrength,
    elements.subjectDetailPreservation,
    elements.backgroundDetailPreservation,
    elements.amp,
    elements.modeSelect,
    elements.pickInput,
    elements.pickOutput,
    ...elements.quickPresetButtons,
    elements.runDenoise
  ]) {
    node.disabled = busy;
  }
  elements.modelSelect.disabled = busy || !elements.modelSelect.value;
  elements.variantStandard.disabled = busy;
  elements.variantLite.disabled = busy;
  elements.cancelDenoise.disabled = !busy;
  elements.saveResult.disabled = busy || !state.hasResult;
  elements.progressMeta.classList.toggle("hidden", !busy);
  elements.progressBar.classList.toggle("hidden", !busy);
  if (!busy) {
    setProgress(0, "Ready");
  }
}

function setProgress(progress, label) {
  const clamped = Math.max(0, Math.min(1, Number(progress) || 0));
  elements.progressBar.firstElementChild.style.width = `${(clamped * 100).toFixed(1)}%`;
  elements.progressLabel.textContent = label || "Working";
  elements.progressPercent.textContent = `${Math.round(clamped * 100)}%`;
}

window.noiseApi.onDenoiseProgress(({ progress, message }) => {
  setProgress(progress, message);
  setStatus(message || `Running ${getModeCopy().noun}...`);
});

function applyModeCopy() {
  const copy = getModeCopy();
  elements.appSubtitle.textContent = copy.subtitle;
  elements.resultLabel.textContent = copy.resultLabel;
  elements.outputTooltipText.textContent = copy.outputTooltip;
  elements.strengthTooltipText.textContent = copy.strengthTooltip;
  elements.runDenoise.textContent = state.mode === "gradient" ? "Run Gradient Removal" : state.mode === "star" ? "Run Star Reducer" : state.mode === "sharpen" ? "Run Sharpen" : "Run";
  elements.gradientOnlySettings?.classList.toggle("hidden", state.mode !== "gradient");
  elements.strength.max = state.mode === "star" ? "3" : "1.5";
  if (Number(elements.strength.value) > Number(elements.strength.max)) {
    elements.strength.value = elements.strength.max;
  }
  updateRangeLabels();
}

function applyModeDefaults() {
  // Shared tiling defaults.
  elements.patchSize.value = "128";
  elements.stride.value = "64";
  elements.batchSize.value = "32";

  if (state.mode === "star") {
    elements.tta.value = "1";
    elements.strength.value = "1.5";
    elements.detailPreservation.value = "0";
    elements.backgroundThreshold.value = "0.12";
    elements.backgroundStrength.value = "1";
    elements.subjectDetailPreservation.value = "0";
    elements.backgroundDetailPreservation.value = "0";
  } else if (state.mode === "sharpen") {
    // The model IS the enhancement: no soft-original blend-back, no background cleanup.
    elements.tta.value = "4";
    elements.strength.value = "1";
    elements.detailPreservation.value = "0";
    elements.backgroundThreshold.value = "0";
    elements.backgroundStrength.value = "1";
    elements.subjectDetailPreservation.value = "0";
    elements.backgroundDetailPreservation.value = "0";
  } else {
    // denoise: restore the standard HTML defaults.
    elements.tta.value = "1";
    elements.strength.value = "1";
    elements.detailPreservation.value = "0.2";
    elements.backgroundThreshold.value = "0.12";
    elements.backgroundStrength.value = "1.20";
    elements.subjectDetailPreservation.value = "0.20";
    elements.backgroundDetailPreservation.value = "0.05";
  }
  updateRangeLabels();
}

function updateRangeLabels() {
  document.getElementById("strengthValue").textContent = Number(elements.strength.value).toFixed(2);
  document.getElementById("detailValue").textContent = Number(elements.detailPreservation.value).toFixed(2);
  document.getElementById("backgroundThresholdValue").textContent = Number(elements.backgroundThreshold.value).toFixed(2);
  document.getElementById("backgroundStrengthValue").textContent = Number(elements.backgroundStrength.value).toFixed(2);
  document.getElementById("subjectDetailValue").textContent = Number(elements.subjectDetailPreservation.value).toFixed(2);
  document.getElementById("backgroundDetailValue").textContent = Number(elements.backgroundDetailPreservation.value).toFixed(2);
  if (elements.gradientBlurSigma && elements.gradientBlurSigmaValue) {
    elements.gradientBlurSigmaValue.textContent = Number(elements.gradientBlurSigma.value).toFixed(1);
  }
}

// ─── Comparison slider ────────────────────────────────────────────────────────

function syncSlider() {
  const canvasW = elements.compareCanvas.offsetWidth;
  elements.denoisedPreview.style.width = `${canvasW}px`;
  elements.compareClip.style.width = `${state.sliderPct}%`;
  elements.compareDivider.style.left = `${state.sliderPct}%`;
}

function initCompareSlider() {
  let isSliding = false;

  elements.compareDivider.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    isSliding = true;
    elements.compareCanvas.style.cursor = "ew-resize";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isSliding) return;
    const rect = elements.compareCanvas.getBoundingClientRect();
    state.sliderPct = Math.max(2, Math.min(98, ((e.clientX - rect.left) / rect.width) * 100));
    syncSlider();
  });

  window.addEventListener("mouseup", () => {
    if (!isSliding) return;
    isSliding = false;
    elements.compareCanvas.style.cursor = "";
  });
}

// ─── Zoom / pan ────────────────────────────────────────────────────────────────

let zoomResetFn = null;

function makeZoomable(canvas, imgs) {
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let isDragging = false;
  let dragStart = { x: 0, y: 0 };

  function applyTransform() {
    const t = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    for (const img of imgs) img.style.transform = t;
  }

  function reset() {
    zoom = 1;
    panX = 0;
    panY = 0;
    isDragging = false;
    for (const img of imgs) img.style.transform = "";
    canvas.style.cursor = "grab";
  }

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.max(0.5, Math.min(40, zoom * factor));
    panX = cx - (cx - panX) * (newZoom / zoom);
    panY = cy - (cy - panY) * (newZoom / zoom);
    zoom = newZoom;
    applyTransform();
    canvas.style.cursor = isDragging ? "grabbing" : "grab";
  }, { passive: false });

  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    isDragging = true;
    dragStart = { x: e.clientX - panX, y: e.clientY - panY };
    canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    panX = e.clientX - dragStart.x;
    panY = e.clientY - dragStart.y;
    applyTransform();
  });

  window.addEventListener("mouseup", () => {
    if (!isDragging) return;
    isDragging = false;
    canvas.style.cursor = "grab";
  });

  return reset;
}

// ─── Preview ──────────────────────────────────────────────────────────────────

function setPreview(target, base64Png) {
  if (zoomResetFn) zoomResetFn();

  if (target === "original") {
    if (!base64Png) {
      elements.originalPreview.removeAttribute("src");
      elements.originalPreview.alt = "";
      elements.originalEmpty.classList.remove("hidden");
    } else {
      elements.originalPreview.src = `data:image/png;base64,${base64Png}`;
      elements.originalPreview.alt = "Original preview";
      elements.originalEmpty.classList.add("hidden");
    }
  } else {
    if (!base64Png) {
      elements.denoisedPreview.removeAttribute("src");
      elements.compareCanvas.classList.remove("has-denoised");
    } else {
      elements.denoisedPreview.src = `data:image/png;base64,${base64Png}`;
      elements.compareCanvas.classList.add("has-denoised");
      syncSlider();
    }
  }
}

function clearDenoisedState() {
  state.hasResult = false;
  state.lastSavedOutputPath = "";
  elements.saveResult.disabled = true;
  setPreview("denoised", "");
}

function applyQuickPreset(presetKey) {
  const preset = QUICK_PRESETS[presetKey];
  if (!preset) {
    return;
  }

  for (const [field, value] of Object.entries(preset.values)) {
    if (elements[field]) {
      elements[field].value = String(value);
    }
  }

  updateRangeLabels();
  setStatus(preset.status);
}

function isLiteModel(model) {
  return model.name.toLowerCase().includes("lite");
}

function filterModelsByVariant(models, variant) {
  if (variant === "lite") return models.filter(isLiteModel);
  return models.filter((m) => !isLiteModel(m));
}

function applyModelVariant(variant) {
  state.modelVariant = variant;

  elements.variantStandard.classList.toggle("is-active", variant === "standard");
  elements.variantStandard.setAttribute("aria-pressed", variant === "standard" ? "true" : "false");
  elements.variantLite.classList.toggle("is-active", variant === "lite");
  elements.variantLite.setAttribute("aria-pressed", variant === "lite" ? "true" : "false");

  const filtered = filterModelsByVariant(state.allModels, variant);
  renderModelOptions(filtered, state.modelPath);

  if (!filtered.length) {
    const msg = variant === "lite"
      ? "No Lite model found. Train one with: python train_lite.py"
      : "No Standard models found in the models folder.";
    setStatus(msg);
  }
}

function renderModelOptions(models, selectedPath = "") {
  elements.modelSelect.innerHTML = "";

  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    elements.modelSelect.append(option);
    elements.modelSelect.disabled = true;
    state.modelPath = "";
    updateBackendBadge();
    return;
  }

  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.name;
    if (model.path === selectedPath) {
      option.selected = true;
    }
    elements.modelSelect.append(option);
  }

  if (!elements.modelSelect.value && models[0]) {
    elements.modelSelect.value = models[0].path;
  }

  elements.modelSelect.disabled = false;
  state.modelPath = elements.modelSelect.value;
  updateBackendBadge();
}

function collectPayload() {
  return {
    mode: state.mode,
    modelPath: state.modelPath,
    inputPath: elements.inputPath.value.trim(),
    outputPath: elements.outputPath.value.trim(),
    patchSize: Number(elements.patchSize.value),
    stride: Number(elements.stride.value),
    tta: Number(elements.tta.value),
    batchSize: Number(elements.batchSize.value),
    amp: elements.amp.checked,
    strength: Number(elements.strength.value),
    detailPreservation: Number(elements.detailPreservation.value),
    backgroundThreshold: Number(elements.backgroundThreshold.value),
    backgroundStrength: Number(elements.backgroundStrength.value),
    subjectDetailPreservation: Number(elements.subjectDetailPreservation.value),
    backgroundDetailPreservation: Number(elements.backgroundDetailPreservation.value),
    gradientBlurSigma: elements.gradientBlurSigma ? Number(elements.gradientBlurSigma.value) : 3.0,
    device: elements.device.value.trim()
  };
}

function validatePayload(payload) {
  if (!payload.modelPath) {
    throw new Error("Select a model from the models folder first.");
  }
  if (!payload.inputPath) {
    throw new Error("Choose an input image first.");
  }
  if (!payload.outputPath) {
    throw new Error("Choose an output path first.");
  }
  if (!Number.isFinite(payload.patchSize) || payload.patchSize <= 0) {
    throw new Error("Patch size must be greater than 0.");
  }
  if (!Number.isFinite(payload.stride) || payload.stride <= 0) {
    throw new Error("Stride must be greater than 0.");
  }
  if (!Number.isFinite(payload.tta) || payload.tta <= 0) {
    throw new Error("TTA must be greater than 0.");
  }
  if (!Number.isFinite(payload.batchSize) || payload.batchSize <= 0) {
    throw new Error("Batch size must be greater than 0.");
  }
  if (!Number.isFinite(payload.backgroundThreshold) || payload.backgroundThreshold < 0 || payload.backgroundThreshold > 1) {
    throw new Error("Background threshold must be between 0 and 1.");
  }
  if (!Number.isFinite(payload.backgroundStrength) || payload.backgroundStrength < 0) {
    throw new Error("Background strength must be 0 or greater.");
  }
  if (!Number.isFinite(payload.subjectDetailPreservation) || payload.subjectDetailPreservation < 0 || payload.subjectDetailPreservation > 1) {
    throw new Error("Subject detail preservation must be between 0 and 1.");
  }
  if (!Number.isFinite(payload.backgroundDetailPreservation) || payload.backgroundDetailPreservation < 0 || payload.backgroundDetailPreservation > 1) {
    throw new Error("Background detail preservation must be between 0 and 1.");
  }
}

elements.pickInput.addEventListener("click", async () => {
  const file = await window.noiseApi.pickInput();
  if (!file) {
    return;
  }
  elements.inputPath.value = file;
  elements.outputPath.value = await window.noiseApi.buildDefaultOutput(file, state.mode);
  clearDenoisedState();
  setStatus("Loading preview...");
  try {
    const result = await window.noiseApi.loadPreview(file);
    setPreview("original", result.preview_base64);
    setStatus(`Image loaded. Adjust settings and run ${getModeCopy().noun} when ready.`);
  } catch (error) {
    setPreview("original", "");
    setStatus(`Failed to open preview: ${error.message}`);
    window.alert(error.message);
  }
});

elements.pickOutput.addEventListener("click", async () => {
  const file = await window.noiseApi.pickOutput(elements.inputPath.value.trim(), elements.outputPath.value.trim());
  if (file) {
    elements.outputPath.value = file;
  }
});

function setActiveQuickPreset(activeButton) {
  for (const b of elements.quickPresetButtons) {
    const on = b === activeButton;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  }
}

for (const button of elements.quickPresetButtons) {
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => {
    const key = button.dataset.preset || "";
    if (!QUICK_PRESETS[key]) {
      return;
    }
    applyQuickPreset(key);
    setActiveQuickPreset(button);
  });
}

elements.variantStandard.addEventListener("click", () => {
  if (state.busy || state.modelVariant === "standard") return;
  applyModelVariant("standard");
});

elements.variantLite.addEventListener("click", () => {
  if (state.busy || state.modelVariant === "lite") return;
  applyModelVariant("lite");
});

elements.runDenoise.addEventListener("click", async () => {
  const payload = collectPayload();
  try {
    validatePayload(payload);
  } catch (error) {
    window.alert(error.message);
    return;
  }

  setBusy(true);
  setProgress(0, "Queued");
  clearDenoisedState();
  setStatus(`Running ${getModeCopy().noun}. Large images can take a while, especially with TTA enabled.`);
  try {
    const result = await window.noiseApi.denoise(payload);
    setPreview("original", result.original_preview_base64);
    setPreview("denoised", result.denoised_preview_base64);
    elements.outputPath.value = result.output_path;
    state.hasResult = true;
    state.lastSavedOutputPath = result.output_path;
    elements.saveResult.disabled = false;
    setStatus(`${getModeCopy().verb[0].toUpperCase()}${getModeCopy().verb.slice(1)} complete. Saved to ${result.output_path}`);
  } catch (error) {
    if (error.message === "Denoising canceled.") {
      setStatus(`${getModeCopy().verb[0].toUpperCase()}${getModeCopy().verb.slice(1)} canceled.`);
    } else {
      setStatus(`${getModeCopy().verb[0].toUpperCase()}${getModeCopy().verb.slice(1)} failed.`);
      window.alert(error.message);
    }
  } finally {
    setBusy(false);
  }
});

elements.cancelDenoise.addEventListener("click", async () => {
  if (!state.busy) {
    return;
  }
  elements.cancelDenoise.disabled = true;
  setStatus("Canceling denoising...");
  try {
    await window.noiseApi.cancelDenoise();
  } catch (error) {
    setStatus(`Cancel request failed: ${error.message}`);
    window.alert(error.message);
  }
});

elements.saveResult.addEventListener("click", async () => {
  if (!state.hasResult) {
    window.alert(`Run ${getModeCopy().noun} first.`);
    return;
  }
  const file = await window.noiseApi.pickOutput(elements.inputPath.value.trim(), elements.outputPath.value.trim());
  if (!file) {
    return;
  }
  try {
    await window.noiseApi.saveCopy(state.lastSavedOutputPath, file);
    elements.outputPath.value = file;
    state.lastSavedOutputPath = file;
    setStatus(`Saved result to ${file}`);
  } catch (error) {
    setStatus("Saving copy failed.");
    window.alert(error.message);
  }
});

elements.strength.addEventListener("input", updateRangeLabels);
elements.detailPreservation.addEventListener("input", updateRangeLabels);
elements.backgroundThreshold.addEventListener("input", updateRangeLabels);
elements.backgroundStrength.addEventListener("input", updateRangeLabels);
elements.subjectDetailPreservation.addEventListener("input", updateRangeLabels);
elements.backgroundDetailPreservation.addEventListener("input", updateRangeLabels);
elements.gradientBlurSigma?.addEventListener("input", updateRangeLabels);
elements.device.addEventListener("input", updateBackendBadge);
elements.modelSelect.addEventListener("change", () => {
  state.modelPath = elements.modelSelect.value;
  updateBackendBadge();
  if (state.modelPath) {
    const backend = inferBackendFromModelPath(state.modelPath);
    setStatus(
      `Model selected: ${elements.modelSelect.options[elements.modelSelect.selectedIndex]?.text || elements.modelSelect.value} (${backend}). PyTorch checkpoints are currently the fastest option in this app.`
    );
  }
});

elements.modeSelect.addEventListener("change", async () => {
  state.mode = elements.modeSelect.value || "denoise";
  applyModeCopy();
  applyModeDefaults();
  clearDenoisedState();
  if (elements.inputPath.value.trim()) {
    elements.outputPath.value = await window.noiseApi.buildDefaultOutput(elements.inputPath.value.trim(), state.mode);
  }
  try {
    const [modelsResult, defaultResult] = await Promise.all([
      window.noiseApi.listModels(state.mode),
      window.noiseApi.getDefaultModel(state.mode)
    ]);
    const models = Array.isArray(modelsResult.models) ? modelsResult.models : [];
    state.allModels = models;
    const filtered = filterModelsByVariant(models, state.modelVariant);
    renderModelOptions(filtered, defaultResult.model_path || "");
    if (state.modelPath) {
      setStatus(`${getModeCopy().resultLabel} mode ready. Choose an image or run with the selected model.`);
    } else if (filtered.length === 0 && state.modelVariant === "lite") {
      setStatus("No Lite model found. Train one with: python train_lite.py");
    } else {
      setStatus(`No ${state.mode === "gradient" ? "gradient-removal" : state.mode === "star" ? "star-reducer" : "denoising"} models were found in the models folder.`);
    }
  } catch (error) {
    state.modelPath = "";
    state.allModels = [];
    renderModelOptions([], "");
    setStatus(`Unable to load models for ${state.mode}: ${error.message}`);
  }
});

// Tooltips: move to document.body so backdrop-filter / overflow do not clip them;
// position in viewport coordinates and clamp to the screen.
const tooltipTriggerByBox = new Map();

function clampTooltipToViewport(box, trigger) {
  const margin = 8;
  const rect = trigger.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const br = box.getBoundingClientRect();
  const bw = br.width;
  const bh = br.height;

  let left = rect.left + rect.width / 2 - bw / 2;
  let top = rect.top - margin - bh;

  if (top < margin) {
    top = rect.bottom + margin;
  }

  left = Math.max(margin, Math.min(left, vw - margin - bw));
  top = Math.max(margin, Math.min(top, vh - margin - bh));

  box.style.left = `${Math.round(left)}px`;
  box.style.top = `${Math.round(top)}px`;
}

function repositionOpenTooltips() {
  tooltipTriggerByBox.forEach((tr, box) => {
    if (box.classList.contains("tooltip-visible")) {
      clampTooltipToViewport(box, tr);
    }
  });
}

document.querySelectorAll(".tooltip").forEach((trigger) => {
  const box = trigger.querySelector(".tooltip-box");
  if (!box) {
    return;
  }

  tooltipTriggerByBox.set(box, trigger);
  let placeholder = null;

  function mountToBody() {
    if (box.parentElement === document.body) {
      return;
    }
    placeholder = document.createComment("tooltip-anchor");
    trigger.insertBefore(placeholder, box);
    document.body.appendChild(box);
  }

  function restoreToTrigger() {
    if (placeholder && placeholder.parentNode === trigger) {
      trigger.insertBefore(box, placeholder);
      placeholder.remove();
    }
    placeholder = null;
  }

  function show() {
    mountToBody();
    box.style.left = "0px";
    box.style.top = "0px";
    box.style.opacity = "0";
    box.classList.add("tooltip-visible");
    requestAnimationFrame(() => {
      clampTooltipToViewport(box, trigger);
      box.style.opacity = "";
    });
  }

  function hide() {
    box.classList.remove("tooltip-visible");
    box.style.left = "";
    box.style.top = "";
    box.style.opacity = "";
    restoreToTrigger();
  }

  trigger.addEventListener("mouseenter", show);
  trigger.addEventListener("focus", show);
  trigger.addEventListener("mouseleave", hide);
  trigger.addEventListener("blur", hide);
});

const controlsScrollArea = document.querySelector(".controls-grid");
if (controlsScrollArea) {
  controlsScrollArea.addEventListener("scroll", repositionOpenTooltips, { passive: true });
}
window.addEventListener("resize", repositionOpenTooltips);

window.addEventListener("DOMContentLoaded", async () => {
  zoomResetFn = makeZoomable(elements.compareCanvas, [
    elements.originalPreview,
    elements.denoisedPreview
  ]);

  initCompareSlider();

  window.addEventListener("resize", () => {
    if (elements.compareCanvas.classList.contains("has-denoised")) {
      syncSlider();
    }
  });

  updateRangeLabels();
  updateBackendBadge();
  applyModeCopy();

  try {
    const [modelsResult, defaultResult] = await Promise.all([
      window.noiseApi.listModels(state.mode),
      window.noiseApi.getDefaultModel(state.mode)
    ]);
    const models = Array.isArray(modelsResult.models) ? modelsResult.models : [];
    state.allModels = models;
    const filtered = filterModelsByVariant(models, state.modelVariant);
    renderModelOptions(filtered, defaultResult.model_path || "");

    if (state.modelPath) {
      setStatus("Model found. Choose an image to start.");
    } else if (filtered.length === 0 && state.modelVariant === "lite") {
      setStatus("No Lite model found. Train one with: python train_lite.py");
    } else {
      state.modelPath = "";
      setStatus("No models were found in the models folder.");
    }
  } catch (error) {
    state.modelPath = "";
    state.allModels = [];
    renderModelOptions([], "");
    setStatus(`Unable to load models from the models folder: ${error.message}`);
  }
});

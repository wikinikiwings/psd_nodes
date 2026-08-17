import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Save PSD (masked layers)
 * ------------------------
 * The python node writes the PSD into ComfyUI's output folder and reports it
 * back as ui.psd. No preview file is produced; this extension only turns that
 * reference into a browser download button on the node.
 */

const NODE_CLASS = "SavePSDLayers";
const IDLE_LABEL = "⬇  download PSD";

function fileUrl(file) {
  const params = new URLSearchParams({
    filename: file.filename,
    subfolder: file.subfolder ?? "",
    type: file.type ?? "output",
  });
  return api.apiURL(`/view?${params.toString()}`);
}

function download(node) {
  const file = node.properties?.lastPsd;
  if (!file?.filename) return;
  const a = document.createElement("a");
  a.href = fileUrl(file);
  a.download = file.filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function setButtonState(node) {
  const btn = node._psdBtn;
  if (!btn) return;
  const file = node.properties?.lastPsd;
  btn.label = file?.filename ? `⬇  download ${file.filename}` : IDLE_LABEL;
  btn.disabled = !file?.filename;
  node.setDirtyCanvas?.(true, true);
}

function ensureButton(node) {
  if (node._psdBtn) return;
  node.properties = node.properties ?? {};
  node._psdBtn = node.addWidget("button", "download_psd", null, () =>
    download(node)
  );
  node._psdBtn.serialize = false;
  node._psdBtn.options = node._psdBtn.options ?? {};
  node._psdBtn.options.serialize = false;
  setButtonState(node);
}

app.registerExtension({
  name: "tapclap.SavePSDLayers",

  nodeCreated(node) {
    if (node?.comfyClass !== NODE_CLASS) return;
    try {
      ensureButton(node);
    } catch (e) {
      console.error("[SavePSDLayers] setup failed", e);
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    // legacy creation path
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try {
        ensureButton(this);
      } catch (e) {
        console.error("[SavePSDLayers] setup failed", e);
      }
      return r;
    };

    // remember the file across workflow reloads
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      try {
        ensureButton(this);
        setButtonState(this);
      } catch (e) {
        /* ignore */
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      try {
        const file = message?.psd?.[0];
        if (file?.filename) {
          this.properties = this.properties ?? {};
          this.properties.lastPsd = file;
          ensureButton(this);
          setButtonState(this);
        }
      } catch (e) {
        console.error("[SavePSDLayers] result handling failed", e);
      }
      return r;
    };
  },
});

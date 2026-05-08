import { useState } from "react";
import { saveGraph, loadGraph, resetGraph } from "../lib/api";

export function PersistencePanel({
  onGraphChanged,
}: {
  onGraphChanged?: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [lastSave, setLastSave] = useState<{
    hkb_bytes: number;
    json_bytes: number;
    timestamp: number;
  } | null>(null);
  const [fromMock, setFromMock] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    const { data, fromMock: mock } = await saveGraph();
    setLastSave({
      hkb_bytes: data.hkb_bytes,
      json_bytes: data.json_bytes,
      timestamp: data.timestamp,
    });
    setFromMock(mock);
    setSaving(false);
  };

  const handleLoad = async () => {
    setLoading(true);
    const { fromMock: mock } = await loadGraph();
    setFromMock(mock);
    setLoading(false);
    onGraphChanged?.();
  };

  const handleReset = async () => {
    setResetting(true);
    const { fromMock: mock } = await resetGraph();
    setFromMock(mock);
    setResetting(false);
    onGraphChanged?.();
  };

  const formatBytes = (bytes: number) => {
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Persistence
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Save, load, reset · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        {/* Buttons */}
        <div className="grid sm:grid-cols-3 gap-2">
          <button
            onClick={handleSave}
            disabled={saving}
            className={[
              "px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
              saving
                ? "bg-konjo-surface/40 text-konjo-fg-muted cursor-not-allowed"
                : "bg-konjo-accent text-konjo-bg hover:bg-konjo-accent/90",
            ].join(" ")}
          >
            {saving ? "Saving…" : "Save Graph"}
          </button>

          <button
            onClick={handleLoad}
            disabled={loading}
            className={[
              "px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
              loading
                ? "bg-konjo-surface/40 text-konjo-fg-muted cursor-not-allowed"
                : "border border-konjo-line text-konjo-fg hover:bg-konjo-line/20",
            ].join(" ")}
          >
            {loading ? "Loading…" : "Load Graph"}
          </button>

          <button
            onClick={handleReset}
            disabled={resetting}
            className={[
              "px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
              resetting
                ? "bg-konjo-surface/40 text-konjo-fg-muted cursor-not-allowed"
                : "border border-konjo-line/60 text-konjo-fg-muted hover:border-konjo-warm hover:text-konjo-warm",
            ].join(" ")}
          >
            {resetting ? "Resetting…" : "Reset"}
          </button>
        </div>

        {/* Last save info */}
        {lastSave && (
          <div className="bg-konjo-surface/60 rounded p-3 space-y-2">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
              Last Save
            </div>
            <div className="flex gap-4 text-[12px]">
              <div>
                <div className="text-konjo-fg-muted">Binary (.hkb)</div>
                <div className="text-konjo-fg font-mono">
                  {formatBytes(lastSave.hkb_bytes)}
                </div>
              </div>
              <div>
                <div className="text-konjo-fg-muted">JSON</div>
                <div className="text-konjo-fg font-mono">
                  {formatBytes(lastSave.json_bytes)}
                </div>
              </div>
              <div>
                <div className="text-konjo-fg-muted">Timestamp</div>
                <div className="text-konjo-fg font-mono">
                  {new Date(lastSave.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info message */}
        <div className="text-konjo-mono text-[11px] text-konjo-fg-muted space-y-1">
          <p>
            • <strong>Save:</strong> Export graph to .hkb (binary) and .json
          </p>
          <p>
            • <strong>Load:</strong> Restore graph from disk backup
          </p>
          <p>
            • <strong>Reset:</strong> Clear all concepts and re-seed with defaults
          </p>
        </div>
      </div>
    </div>
  );
}

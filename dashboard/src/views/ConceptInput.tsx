import { useState } from "react";
import { encodeConcept, addConcept } from "../lib/api";
import type { EncodeResult } from "../lib/api";

export function ConceptInput({
  onConceptAdded,
}: {
  onConceptAdded?: () => void;
}) {
  const [input, setInput] = useState("");
  const [encoded, setEncoded] = useState<EncodeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);
  const [added, setAdded] = useState(false);

  const handleEncode = async () => {
    if (!input.trim()) return;
    setLoading(true);
    const { data, fromMock: mock } = await encodeConcept(input);
    setEncoded(data);
    setFromMock(mock);
    setAdded(false);
    setLoading(false);
  };

  const handleAdd = async () => {
    if (!input.trim() || !encoded) return;
    setLoading(true);
    const { fromMock: mock } = await addConcept(input);
    setFromMock(mock);
    setAdded(true);
    setLoading(false);
    setTimeout(() => {
      setInput("");
      setEncoded(null);
      setAdded(false);
      onConceptAdded?.();
    }, 1500);
  };

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Add Concept
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Encode and store · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div>
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2 block">
            Concept text
          </label>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !encoded) handleEncode();
            }}
            placeholder="e.g., 'neural networks'"
            className="w-full px-3 py-2 rounded-konjo bg-konjo-surface border border-konjo-line text-konjo-fg placeholder:text-konjo-fg-muted text-[13px] focus:outline-none focus:ring-2 focus:ring-konjo-accent"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleEncode}
            disabled={!input.trim() || loading || !!encoded}
            className={[
              "px-3 py-2 rounded-konjo text-[12px] font-konjo-mono uppercase transition-colors",
              !input.trim() || loading || !!encoded
                ? "bg-konjo-line text-konjo-fg-muted cursor-not-allowed opacity-50"
                : "bg-konjo-accent text-konjo-bg hover:brightness-110",
            ].join(" ")}
          >
            {loading ? "Encoding…" : "Encode"}
          </button>

          {encoded && (
            <button
              onClick={handleAdd}
              disabled={loading || added}
              className={[
                "px-3 py-2 rounded-konjo text-[12px] font-konjo-mono uppercase transition-colors",
                loading || added
                  ? "bg-konjo-line text-konjo-fg-muted cursor-not-allowed opacity-50"
                  : "bg-konjo-good text-konjo-bg hover:brightness-110",
              ].join(" ")}
            >
              {added ? "Added ✓" : "Add to Graph"}
            </button>
          )}
        </div>

        {encoded && (
          <div className="rounded-konjo p-3 space-y-2 bg-konjo-surface-2">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
              Vector Preview
            </div>
            <div className="flex gap-1">
              {encoded.vector_preview.map((val: number, i: number) => (
                <div
                  key={i}
                  className="text-[11px] text-konjo-fg font-mono"
                >
                  {val.toFixed(3)}
                </div>
              ))}
              …
            </div>
            <div className="flex justify-between text-[12px] text-konjo-fg-muted">
              <span>Norm: {encoded.norm.toFixed(3)}</span>
              <span>{encoded.latency_ms.toFixed(1)}ms</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

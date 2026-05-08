import { useState } from "react";
import { queryGraph } from "../lib/api";
import type { QueryResult } from "../lib/api";

export function QueryPanel({
  cosineWeight,
  decayHalfLife,
}: {
  cosineWeight: number;
  decayHalfLife: number;
}) {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fromMock, setFromMock] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    const { data, fromMock: mock } = await queryGraph(
      query,
      cosineWeight,
      decayHalfLife
    );
    setResult(data);
    setFromMock(mock);
    setLoading(false);
  };

  const sampleQueries = ["temporal decay", "semantic matching", "memory retrieval"];

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Query Graph
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Search with decay · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        {/* Input */}
        <div>
          <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2 block">
            Search query
          </label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleQuery();
            }}
            placeholder="e.g., 'find concepts related to…'"
            className="w-full px-3 py-2 rounded-konjo bg-konjo-surface/40 border border-konjo-line text-konjo-fg placeholder-konjo-fg-muted text-[13px] focus:outline-none focus:ring-2 focus:ring-konjo-accent mb-3"
          />
          <div className="flex flex-wrap gap-1">
            {sampleQueries.map((sq) => (
              <button
                key={sq}
                onClick={() => setQuery(sq)}
                className="text-[11px] px-2 py-1 rounded bg-konjo-line/40 text-konjo-fg-muted hover:text-konjo-fg transition-colors"
              >
                {sq}
              </button>
            ))}
          </div>
        </div>

        {/* Search button */}
        <button
          onClick={handleQuery}
          disabled={!query.trim() || loading}
          className={[
            "w-full px-3 py-2 rounded-konjo text-[12px] font-mono uppercase transition-colors",
            !query.trim() || loading
              ? "bg-konjo-surface/40 text-konjo-fg-muted cursor-not-allowed"
              : "bg-konjo-accent text-konjo-bg hover:bg-konjo-accent/90",
          ].join(" ")}
        >
          {loading ? "Searching…" : "Search"}
        </button>

        {/* Results */}
        {result && (
          <div className="space-y-2">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
              Top Results · {result.elapsed_ms.toFixed(1)}ms
            </div>

            <div className="space-y-2">
              {result.matches.map((match: any) => (
                <div
                  key={match.rank}
                  className="bg-konjo-surface/60 rounded p-3"
                >
                  <div className="flex items-baseline justify-between gap-2 mb-1">
                    <div className="text-konjo-fg font-medium text-[13px]">
                      {match.rank}. {match.concept}
                    </div>
                    <div
                      className="text-konjo-accent font-mono text-[11px]"
                      style={{
                        width: `${match.composite_score * 100}px`,
                        display: "inline-block",
                      }}
                    >
                      {(match.composite_score * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div className="text-konjo-fg-muted text-[11px] flex justify-between">
                    <span>
                      cosine: {match.cosine_score.toFixed(3)} · decay:{" "}
                      {match.decay_weight.toFixed(3)}
                    </span>
                    <span>age: {match.age_steps}s</span>
                  </div>

                  {/* Score bar */}
                  <div className="mt-2 h-1.5 bg-konjo-line/30 rounded overflow-hidden">
                    <div
                      className="h-full bg-konjo-accent"
                      style={{ width: `${match.composite_score * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="text-konjo-mono text-[11px] text-konjo-fg-muted animate-pulse">
            querying…
          </div>
        )}
      </div>
    </div>
  );
}

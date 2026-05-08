import { useEffect, useState, useRef } from "react";
import { fetchGraph } from "../lib/api";
import type { GraphResponse } from "../lib/api";

export function MemoryGraph({
  refreshTrigger,
}: {
  refreshTrigger?: number;
}) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [fromMock, setFromMock] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const { data, fromMock: mock } = await fetchGraph();
      setGraph(data);
      setFromMock(mock);
      setLoading(false);
    })();
  }, [refreshTrigger]);

  if (loading) {
    return (
      <div className="space-y-3">
        <header>
          <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
            Memory Graph
          </h2>
          <p className="text-konjo-fg-muted text-[13px] mt-1">
            Concept relationships
          </p>
        </header>
        <div className="glass-konjo rounded-konjo-lg p-5 h-64 flex items-center justify-center">
          <p className="text-konjo-fg-muted text-konjo-mono text-[12px]">loading graph…</p>
        </div>
      </div>
    );
  }

  if (!graph || graph.nodes.length === 0) {
    return null;
  }

  // Calculate node positions in a circle
  const centerX = 250;
  const centerY = 200;
  const radius = 150;
  const positions = graph.nodes.map((_node, idx) => {
    const angle = (idx / graph.nodes.length) * Math.PI * 2;
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  });

  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-konjo-display text-konjo-fg" style={{ fontSize: 20, fontWeight: 600 }}>
          Memory Graph
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Concept relationships · <span className="text-konjo-fg">{fromMock ? "mock" : "live"}</span>
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        {/* SVG Graph */}
        <svg
          ref={svgRef}
          width="100%"
          height="400"
          viewBox="0 0 500 400"
          className="bg-konjo-surface/40 rounded border border-konjo-line/60"
        >
          {/* Edges */}
          {graph.nodes.map((_node: any, i: number) =>
            graph.nodes.map((_target: any, j: number) => {
              if (i >= j) return null; // Only draw each edge once
              const sim = graph.similarities[i][j];
              if (sim < 0.3) return null; // Skip weak connections

              const opacity = Math.max(0.1, sim - 0.2);
              const strokeWidth = Math.max(0.5, (sim - 0.2) * 2);
              const hoveredIdx = hoveredNode
                ? graph.nodes.findIndex((n: any) => n.id === hoveredNode)
                : -1;
              const isRelevant =
                hoveredIdx === -1 ||
                hoveredIdx === i ||
                hoveredIdx === j;

              return (
                <line
                  key={`edge-${i}-${j}`}
                  x1={positions[i].x}
                  y1={positions[i].y}
                  x2={positions[j].x}
                  y2={positions[j].y}
                  stroke="var(--color-konjo-accent)"
                  strokeWidth={strokeWidth}
                  opacity={isRelevant ? opacity : opacity * 0.2}
                  pointerEvents="none"
                />
              );
            })
          )}

          {/* Nodes */}
          {graph.nodes.map((node: any, idx: number) => {
            const isHovered = hoveredNode === node.id;
            const hoveredIdx = hoveredNode
              ? graph.nodes.findIndex((n: any) => n.id === hoveredNode)
              : -1;
            const isConnected =
              hoveredIdx >= 0 &&
              (hoveredIdx === idx ||
                graph.similarities[hoveredIdx][idx] > 0.4);
            const isVisible =
              hoveredNode === null || isHovered || isConnected;

            return (
              <g
                key={node.id}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                style={{ cursor: "pointer" }}
              >
                {/* Circle */}
                <circle
                  cx={positions[idx].x}
                  cy={positions[idx].y}
                  r={isHovered ? 18 : 12}
                  fill={
                    isHovered
                      ? "var(--color-konjo-accent)"
                      : "var(--color-konjo-good)"
                  }
                  opacity={isVisible ? 1 : 0.15}
                  className="transition-all duration-200"
                />

                {/* Label */}
                <text
                  x={positions[idx].x}
                  y={positions[idx].y + 30}
                  textAnchor="middle"
                  className="text-konjo-mono text-[11px]"
                  fill="var(--color-konjo-fg)"
                  opacity={isVisible ? 1 : 0.2}
                  pointerEvents="none"
                >
                  {node.concept}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Node list */}
        {hoveredNode && (
          <div className="bg-konjo-surface/60 rounded p-3 space-y-2">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
              Node Details
            </div>
            <div className="text-konjo-fg text-[12px] space-y-1">
              <div>
                <strong>Concept:</strong>{" "}
                {graph.nodes.find((n: any) => n.id === hoveredNode)?.concept}
              </div>
              <div>
                <strong>Norm:</strong>{" "}
                {graph.nodes.find((n: any) => n.id === hoveredNode)?.norm.toFixed(3)}
              </div>
              <div>
                <strong>Added:</strong>{" "}
                {graph.nodes.find((n: any) => n.id === hoveredNode)?.added_at}
              </div>
            </div>
          </div>
        )}

        <div className="text-konjo-mono text-[11px] text-konjo-fg-muted">
          {graph.nodes.length} concepts · hover to explore connections
        </div>
      </div>
    </div>
  );
}

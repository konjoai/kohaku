import { useState, useMemo } from "react";

export function DecayCurve({
  onParamsChange,
}: {
  onParamsChange?: (cosine_weight: number, decay_half_life: number) => void;
}) {
  const [cosineWeight, setCosineWeight] = useState(1.0);
  const [decayHalfLife, setDecayHalfLife] = useState(500);

  const handleCosineChange = (val: number) => {
    setCosineWeight(val);
    onParamsChange?.(val, decayHalfLife);
  };

  const handleDecayChange = (val: number) => {
    setDecayHalfLife(val);
    onParamsChange?.(cosineWeight, val);
  };

  // Generate decay curve points for SVG
  const curvePoints = useMemo(() => {
    const points: Array<[number, number]> = [];
    const maxAge = decayHalfLife * 3;
    const step = maxAge / 100;

    for (let i = 0; i <= maxAge; i += step) {
      const decay = Math.pow(0.5, i / decayHalfLife);
      points.push([i, decay]);
    }
    return points;
  }, [decayHalfLife]);

  // Scale to SVG viewport (300x150)
  const svgWidth = 300;
  const svgHeight = 150;
  const maxAge = decayHalfLife * 3;

  const pathData = curvePoints
    .map((point, idx) => {
      const x = (point[0] / maxAge) * svgWidth;
      const y = svgHeight - point[1] * svgHeight;
      return `${idx === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  return (
    <div className="space-y-3">
      <header>
        <h2
          className="text-konjo-display text-konjo-fg"
          style={{ fontSize: 20, fontWeight: 600 }}
        >
          Decay Tuning
        </h2>
        <p className="text-konjo-fg-muted text-[13px] mt-1">
          Adjust temporal weight decay
        </p>
      </header>

      <div className="glass-konjo rounded-konjo-lg p-5 space-y-4">
        <div className="flex items-start gap-6">
          {/* Decay Curve */}
          <div className="flex-shrink-0">
            <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2">
              Decay Function
            </div>
            <svg
              width={svgWidth}
              height={svgHeight}
              className="bg-konjo-surface/40 rounded border border-konjo-line/60"
            >
              {/* Grid lines */}
              <line
                x1="0"
                y1={svgHeight}
                x2={svgWidth}
                y2={svgHeight}
                stroke="var(--color-konjo-line)"
                strokeWidth="1"
                opacity="0.3"
              />
              <line
                x1="0"
                y1={svgHeight / 2}
                x2={svgWidth}
                y2={svgHeight / 2}
                stroke="var(--color-konjo-line)"
                strokeWidth="1"
                opacity="0.2"
              />

              {/* Decay curve */}
              <path
                d={pathData}
                fill="none"
                stroke="var(--color-konjo-accent)"
                strokeWidth="2"
              />

              {/* Half-life marker */}
              <line
                x1={(decayHalfLife / maxAge) * svgWidth}
                y1="0"
                x2={(decayHalfLife / maxAge) * svgWidth}
                y2={svgHeight}
                stroke="var(--color-konjo-warm)"
                strokeWidth="1"
                strokeDasharray="4 4"
                opacity="0.5"
              />

              {/* Labels */}
              <text
                x={svgWidth - 4}
                y={10}
                textAnchor="end"
                className="text-konjo-mono text-[9px]"
                fill="var(--color-konjo-fg-muted)"
              >
                decay(t)
              </text>
              <text
                x={4}
                y={svgHeight - 2}
                className="text-konjo-mono text-[9px]"
                fill="var(--color-konjo-fg-muted)"
              >
                t
              </text>
            </svg>
          </div>

          {/* Controls */}
          <div className="flex-1 space-y-4 min-w-0">
            <div>
              <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2 block">
                Cosine Weight
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={cosineWeight}
                  onChange={(e) => handleCosineChange(parseFloat(e.target.value))}
                  className="flex-1"
                />
                <span className="text-konjo-fg font-mono text-[12px] min-w-[40px]">
                  {cosineWeight.toFixed(1)}
                </span>
              </div>
            </div>

            <div>
              <label className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted mb-2 block">
                Decay Half-life
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="50"
                  value={decayHalfLife}
                  onChange={(e) => handleDecayChange(parseFloat(e.target.value))}
                  className="flex-1"
                />
                <span className="text-konjo-fg font-mono text-[12px] min-w-[50px]">
                  {decayHalfLife}s
                </span>
              </div>
            </div>

            <div className="bg-konjo-surface/60 rounded p-3 space-y-1">
              <div className="text-konjo-mono uppercase tracking-[0.16em] text-[10px] text-konjo-fg-muted">
                Formula
              </div>
              <div className="text-konjo-fg text-[12px] font-mono">
                decay(t) = 0.5^(t/{decayHalfLife})
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

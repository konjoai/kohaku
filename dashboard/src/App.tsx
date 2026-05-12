import { useState } from "react";
import { MemoryGraph } from "./views/MemoryGraph";
import { ConceptInput } from "./views/ConceptInput";
import { DecayCurve } from "./views/DecayCurve";
import { QueryPanel } from "./views/QueryPanel";
import { PersistencePanel } from "./views/PersistencePanel";
import { MetaInspector } from "./views/MetaInspector";

export default function App() {
  const [cosineWeight, setCosineWeight] = useState(1.0);
  const [decayHalfLife, setDecayHalfLife] = useState(500);
  const [refreshGraph, setRefreshGraph] = useState(0);

  return (
    <div className="min-h-full bg-konjo-bg text-konjo-fg aurora-konjo">
      <div className="aurora-konjo-bg" />

      <div className="relative z-10 max-w-[1200px] mx-auto px-8 pb-16">
        <Hero />

        <MetaInspector />

        <div className="space-y-6 mt-8">
          <MemoryGraph refreshTrigger={refreshGraph} />

          <section className="grid lg:grid-cols-2 gap-4">
            <ConceptInput onConceptAdded={() => setRefreshGraph((n) => n + 1)} />
            <DecayCurve
              onParamsChange={(cw, dhl) => {
                setCosineWeight(cw);
                setDecayHalfLife(dhl);
              }}
            />
          </section>

          <QueryPanel cosineWeight={cosineWeight} decayHalfLife={decayHalfLife} />

          <PersistencePanel onGraphChanged={() => setRefreshGraph((n) => n + 1)} />

          <Footer />
        </div>
      </div>
    </div>
  );
}

function Hero() {
  return (
    <section className="text-center pt-16 pb-10">
      <p className="text-konjo-mono text-konjo-violet uppercase tracking-[0.32em] text-[11px] mb-5">
        kohaku · 記憶 · episodic memory · 概念
      </p>

      <h1 className="text-konjo-display text-[52px] leading-[1.05] tracking-tight max-w-3xl mx-auto mb-5 text-konjo-fg">
        Concepts,{" "}
        <span className="text-konjo-accent">remembered</span>.
      </h1>

      <p className="text-konjo-fg-muted text-[16px] max-w-xl mx-auto leading-relaxed mb-8">
        Hypervector-based semantic memory with temporal decay. Store knowledge,
        retrieve relationships, watch your memory age gracefully.
      </p>

      <div className="flex gap-2 justify-center flex-wrap">
        {["HDC · Rust", "Temporal Decay", "PyO3 Bindings", "v0.4.0"].map((tag) => (
          <span
            key={tag}
            className="text-konjo-mono text-[11px] text-konjo-fg-muted px-3 py-1 rounded-full border border-konjo-line bg-konjo-surface"
          >
            {tag}
          </span>
        ))}
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-mono text-konjo-fg-muted text-[12px]">
      <div className="flex flex-wrap gap-4 justify-between items-baseline">
        <span>
          built on{" "}
          <span className="text-konjo-fg">@konjoai/ui</span>
          {" · "}
          <span className="text-konjo-fg">/api/graph</span>
          {" · "}
          <span className="text-konjo-fg">/api/query</span>
          {" · "}
          <span className="text-konjo-fg">/api/encode</span>
        </span>
        <span className="text-konjo-fg-faint">
          part of the KonjoAI portfolio · squish · kyro · miru · kohaku · kairu
          · toki · squash
        </span>
      </div>
    </footer>
  );
}

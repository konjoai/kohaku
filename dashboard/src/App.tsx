import { useState } from "react";
import { KonjoApp } from "@konjoai/ui";
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

  const handleGraphChanged = () => {
    setRefreshGraph((prev) => prev + 1);
  };

  return (
    <KonjoApp
      product="kohaku"
      tagline="Hypervector Memory · Semantic Storage · Temporal Decay"
      status={{ label: "ready", severity: "ok" }}
    >
      <Hero />

      <div className="space-y-6 mt-10">
        <MemoryGraph refreshTrigger={refreshGraph} />

        <section className="grid lg:grid-cols-2 gap-4">
          <ConceptInput onConceptAdded={handleGraphChanged} />
          <DecayCurve
            onParamsChange={(cw, dhl) => {
              setCosineWeight(cw);
              setDecayHalfLife(dhl);
            }}
          />
        </section>

        <QueryPanel
          cosineWeight={cosineWeight}
          decayHalfLife={decayHalfLife}
        />

        <PersistencePanel onGraphChanged={handleGraphChanged} />

        <MetaInspector />

        <Footer />
      </div>
    </KonjoApp>
  );
}

function Hero() {
  return (
    <section className="text-center pt-6 pb-2">
      <p
        className="text-konjo-mono uppercase tracking-[0.32em] text-konjo-violet"
        style={{ fontSize: 11 }}
      >
        kohaku · 记忆 · memory · 概念
      </p>
      <h1
        className="text-konjo-display text-konjo-fg mt-4 mx-auto"
        style={{
          fontSize: 52,
          fontWeight: 600,
          letterSpacing: "-0.025em",
          maxWidth: 920,
          lineHeight: 1.05,
        }}
      >
        Concepts,{" "}
        <span style={{ color: "var(--color-konjo-accent)" }}>remembered</span>.
      </h1>
      <p
        className="text-konjo-fg-muted mt-5 mx-auto"
        style={{ fontSize: 16, maxWidth: 640, lineHeight: 1.55 }}
      >
        Hypervector-based semantic memory with temporal decay. Store knowledge,
        retrieve relationships, watch your memory age gracefully.
      </p>
    </section>
  );
}

function Footer() {
  return (
    <footer
      className="mt-16 pt-8 border-t border-konjo-line/60 text-konjo-fg-muted text-konjo-mono"
      style={{ fontSize: 12 }}
    >
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

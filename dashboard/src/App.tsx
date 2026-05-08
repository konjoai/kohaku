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

  const handleGraphChanged = () => {
    setRefreshGraph((prev) => prev + 1);
  };

  return (
    <div style={{ width: "100%", maxWidth: "1200px", margin: "0 auto", padding: "2rem", textAlign: "left" }}>
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
    </div>
  );
}

function Hero() {
  return (
    <section style={{ textAlign: "center", paddingTop: "24px", paddingBottom: "8px" }}>
      <p style={{ fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.32em", color: "var(--color-konjo-violet)" }}>
        kohaku · 记忆 · memory · 概念
      </p>
      <h1 style={{ fontSize: "52px", fontWeight: 600, letterSpacing: "-0.025em", maxWidth: "920px", lineHeight: 1.05, margin: "16px auto", color: "var(--color-konjo-fg)" }}>
        Concepts,{" "}
        <span style={{ color: "var(--color-konjo-accent)" }}>remembered</span>.
      </h1>
      <p style={{ fontSize: "16px", maxWidth: "640px", lineHeight: 1.55, margin: "20px auto", color: "var(--color-konjo-fg-muted)" }}>
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

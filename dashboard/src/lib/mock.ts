import type {
  HealthResponse,
  GraphResponse,
  Node,
  EncodeResult,
  ConceptUpdate,
  QueryResult,
  QueryMatch,
  SaveResult,
  LoadResult,
} from "./types";

export const mockHealth: HealthResponse = {
  version: "1.0.0",
  dims: 768,
  num_concepts: 7,
  internal_clock: 1200,
};

const seedConcepts: Node[] = [
  {
    id: "c0",
    concept: "embeddings",
    vector_preview: [0.12, -0.45, 0.78, 0.34, -0.21],
    norm: 0.987,
    added_at: 100,
  },
  {
    id: "c1",
    concept: "quantization",
    vector_preview: [0.23, 0.56, -0.34, 0.12, 0.89],
    norm: 0.956,
    added_at: 200,
  },
  {
    id: "c2",
    concept: "inference",
    vector_preview: [0.45, -0.12, 0.67, 0.23, -0.45],
    norm: 0.934,
    added_at: 300,
  },
  {
    id: "c3",
    concept: "hypervectors",
    vector_preview: [-0.34, 0.78, 0.12, -0.56, 0.23],
    norm: 0.912,
    added_at: 400,
  },
  {
    id: "c4",
    concept: "semantic space",
    vector_preview: [0.67, 0.23, -0.45, 0.78, 0.12],
    norm: 0.898,
    added_at: 500,
  },
  {
    id: "c5",
    concept: "memory",
    vector_preview: [-0.12, 0.34, 0.56, -0.78, 0.45],
    norm: 0.876,
    added_at: 600,
  },
  {
    id: "c6",
    concept: "learning",
    vector_preview: [0.34, -0.67, 0.23, 0.12, -0.89],
    norm: 0.854,
    added_at: 700,
  },
];

// 7x7 similarity matrix (symmetric)
const seedSimilarities: number[][] = [
  [1.0, 0.72, 0.56, 0.45, 0.68, 0.34, 0.23],
  [0.72, 1.0, 0.48, 0.62, 0.44, 0.58, 0.37],
  [0.56, 0.48, 1.0, 0.51, 0.63, 0.29, 0.46],
  [0.45, 0.62, 0.51, 1.0, 0.55, 0.71, 0.38],
  [0.68, 0.44, 0.63, 0.55, 1.0, 0.42, 0.59],
  [0.34, 0.58, 0.29, 0.71, 0.42, 1.0, 0.64],
  [0.23, 0.37, 0.46, 0.38, 0.59, 0.64, 1.0],
];

export const buildMockGraph = (): GraphResponse => ({
  nodes: seedConcepts,
  similarities: seedSimilarities,
});

export const buildMockEncode = (_concept: string): EncodeResult => ({
  vector_preview: [
    Math.random() * 2 - 1,
    Math.random() * 2 - 1,
    Math.random() * 2 - 1,
    Math.random() * 2 - 1,
    Math.random() * 2 - 1,
  ],
  norm: 0.85 + Math.random() * 0.15,
  latency_ms: 2.3 + Math.random() * 1.5,
});

export const buildMockConceptUpdate = (concept: string): ConceptUpdate => {
  const graph = buildMockGraph();
  return {
    concept,
    graph,
    added_at: Date.now(),
  };
};

export const buildMockQueryResult = (
  query: string,
  currentTime: number,
  decay_half_life: number
): QueryResult => {
  const now = currentTime;
  const matches: QueryMatch[] = seedConcepts
    .map((node, idx) => {
      const cosine = 0.45 + Math.random() * 0.5;
      const age_steps = now - node.added_at;
      const decay_weight = Math.pow(0.5, age_steps / decay_half_life);
      const composite_score = cosine * decay_weight;
      return {
        rank: idx + 1,
        concept: node.concept,
        cosine_score: cosine,
        decay_weight,
        composite_score,
        age_steps,
      };
    })
    .sort((a, b) => b.composite_score - a.composite_score)
    .map((m, idx) => ({ ...m, rank: idx + 1 }));

  return {
    query,
    matches: matches.slice(0, 5),
    elapsed_ms: 3.2 + Math.random() * 2.8,
  };
};

export const buildMockSaveResult = (): SaveResult => ({
  hkb_path: "/tmp/kohaku_backup.hkb",
  json_path: "/tmp/kohaku_backup.json",
  hkb_bytes: 125000 + Math.floor(Math.random() * 50000),
  json_bytes: 45000 + Math.floor(Math.random() * 20000),
  timestamp: Date.now(),
});

export const buildMockLoadResult = (): LoadResult => ({
  graph: buildMockGraph(),
  loaded_concepts: 7,
  timestamp: Date.now(),
});

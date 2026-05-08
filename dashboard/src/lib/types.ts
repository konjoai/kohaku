export interface HealthResponse {
  version: string;
  dims: number;
  num_concepts: number;
  internal_clock: number;
}

export interface Node {
  id: string;
  concept: string;
  vector_preview: number[];
  norm: number;
  added_at: number;
}

export interface GraphResponse {
  nodes: Node[];
  similarities: number[][]; // adjacency matrix
}

export interface EncodeResult {
  vector_preview: number[];
  norm: number;
  latency_ms: number;
}

export interface ConceptUpdate {
  concept: string;
  graph: GraphResponse;
  added_at: number;
}

export interface QueryMatch {
  rank: number;
  concept: string;
  cosine_score: number;
  decay_weight: number;
  composite_score: number;
  age_steps: number;
}

export interface QueryResult {
  query: string;
  matches: QueryMatch[];
  elapsed_ms: number;
}

export interface SaveResult {
  hkb_path: string;
  json_path: string;
  hkb_bytes: number;
  json_bytes: number;
  timestamp: number;
}

export interface LoadResult {
  graph: GraphResponse;
  loaded_concepts: number;
  timestamp: number;
}

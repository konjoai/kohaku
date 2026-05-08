import type {
  HealthResponse,
  GraphResponse,
  EncodeResult,
  ConceptUpdate,
  QueryResult,
  SaveResult,
  LoadResult,
} from "./types";
import {
  mockHealth,
  buildMockGraph,
  buildMockEncode,
  buildMockConceptUpdate,
  buildMockQueryResult,
  buildMockSaveResult,
  buildMockLoadResult,
} from "./mock";

export type {
  HealthResponse,
  GraphResponse,
  EncodeResult,
  ConceptUpdate,
  QueryResult,
  SaveResult,
  LoadResult,
};

export interface ApiResponse<T> {
  data: T;
  fromMock: boolean;
}

export async function fetchHealth(): Promise<ApiResponse<HealthResponse>> {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Failed to fetch health");
    const data: HealthResponse = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: mockHealth, fromMock: true };
  }
}

export async function fetchGraph(): Promise<ApiResponse<GraphResponse>> {
  try {
    const response = await fetch("/api/graph");
    if (!response.ok) throw new Error("Failed to fetch graph");
    const data: GraphResponse = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockGraph(), fromMock: true };
  }
}

export async function encodeConcept(
  concept: string
): Promise<ApiResponse<EncodeResult>> {
  try {
    const response = await fetch("/api/encode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ concept }),
    });
    if (!response.ok) throw new Error("Failed to encode concept");
    const data: EncodeResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockEncode(concept), fromMock: true };
  }
}

export async function addConcept(
  concept: string
): Promise<ApiResponse<ConceptUpdate>> {
  try {
    const response = await fetch("/api/add-concept", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ concept }),
    });
    if (!response.ok) throw new Error("Failed to add concept");
    const data: ConceptUpdate = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockConceptUpdate(concept), fromMock: true };
  }
}

export async function queryGraph(
  query: string,
  cosine_weight: number,
  decay_half_life: number
): Promise<ApiResponse<QueryResult>> {
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, cosine_weight, decay_half_life }),
    });
    if (!response.ok) throw new Error("Failed to query graph");
    const data: QueryResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return {
      data: buildMockQueryResult(query, Date.now(), decay_half_life),
      fromMock: true,
    };
  }
}

export async function saveGraph(): Promise<ApiResponse<SaveResult>> {
  try {
    const response = await fetch("/api/save", { method: "POST" });
    if (!response.ok) throw new Error("Failed to save graph");
    const data: SaveResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockSaveResult(), fromMock: true };
  }
}

export async function loadGraph(): Promise<ApiResponse<LoadResult>> {
  try {
    const response = await fetch("/api/load", { method: "POST" });
    if (!response.ok) throw new Error("Failed to load graph");
    const data: LoadResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockLoadResult(), fromMock: true };
  }
}

export async function resetGraph(): Promise<ApiResponse<GraphResponse>> {
  try {
    const response = await fetch("/api/reset", { method: "POST" });
    if (!response.ok) throw new Error("Failed to reset graph");
    const data: GraphResponse = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockGraph(), fromMock: true };
  }
}

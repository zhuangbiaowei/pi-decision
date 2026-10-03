import type { EvaluatorClient } from "./types.js";
import { JevClient } from "./jev.js";
import { LlamaCppClient } from "./llamacpp.js";

export type EvaluatorProvider = "typesafe" | "llamacpp";

const PROVIDER_ALIASES: Record<string, EvaluatorProvider> = {
  typesafe: "typesafe",
  jev: "typesafe",
  "typesafe-jev": "typesafe",
  llamacpp: "llamacpp",
  "llama-cpp": "llamacpp",
  llama: "llamacpp",
  openai: "llamacpp",
};

export function resolveProvider(): EvaluatorProvider {
  const raw = process.env.PI_JEV_PROVIDER?.trim().toLowerCase();
  if (!raw) return "typesafe";
  return PROVIDER_ALIASES[raw] ?? "typesafe";
}

/** Build the evaluator for the configured provider (default: TypeSafe Jev). */
export function createEvaluator(): EvaluatorClient {
  return resolveProvider() === "llamacpp" ? new LlamaCppClient() : new JevClient();
}

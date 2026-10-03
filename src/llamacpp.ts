import type {
  EvaluatorClient,
  JevAnswerResult,
  JevEvaluationRequest,
  JevEvaluationResponse,
  JevSessionStats,
  QuestionConfig,
} from "./types.js";

/**
 * Alternative evaluator for an OpenAI-compatible llama.cpp server serving a
 * System One style decision model (e.g. StartLux-Decision).
 *
 * Selected with `PI_JEV_PROVIDER=llamacpp`. This provider does not use the
 * TypeSafe SDK at all; it translates the same `evaluate()` contract into a
 * `/v1/chat/completions` request asking the model to answer every question
 * with a single JSON object, then normalizes the result back into the
 * `JevEvaluationResponse` shape the rest of the extension already consumes.
 *
 * The model under test is a *reasoning* model: its final answer may arrive in
 * `choices[0].message.content` OR `choices[0].message.reasoning_content`, and
 * the choice is not consistent between requests. We therefore parse JSON out
 * of both fields.
 */

export const LLAMACPP_DEFAULT_MODEL = "StartLux-Decision-9B-Q8_0";

const SYSTEM_PROMPT = [
  "You are a decision engine. Given STATE and a list of QUESTIONS, answer each question.",
  "Reply with ONLY one JSON object, no prose, no markdown code fences, in exactly this shape:",
  '{"answers":{"<question_id>": <answer>}}',
  "Answer types:",
  '- "noul": a float probability between 0.0 and 1.0.',
  '- "choice": the chosen option key as a string (one of the listed keys).',
  '- "score": a float score.',
].join("\n");

export function resolveLlamacppBaseURL(): string | null {
  return (
    process.env.PI_LLAMACPP_BASE_URL?.trim() ||
    process.env.LLAMACPP_BASE_URL?.trim() ||
    null
  );
}

export function resolveLlamacppModel(): string {
  return (
    process.env.PI_LLAMACPP_MODEL?.trim() ||
    process.env.LLAMACPP_MODEL?.trim() ||
    LLAMACPP_DEFAULT_MODEL
  );
}

/** Pull the first `{...}` JSON object out of text that may include prose. */
export function extractJsonObject(text: string): unknown {
  if (!text) return null;
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end <= start) return null;
  try {
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
}

/**
 * The decision model occasionally emits malformed JSON on larger prompts
 * (e.g. an unbalanced quote: `"tool_search:0.0, "ls":0.0}`), which strict
 * JSON.parse rejects. Fall back to scanning key/value pairs so a single
 * broken quote does not lose the whole answer set.
 */
export function parseAnswerMap(text: string): Record<string, unknown> | null {
  const obj = extractJsonObject(text);
  if (obj && typeof obj === "object") return obj as Record<string, unknown>;

  const pairs: Record<string, unknown> = {};
  const re = /"?([A-Za-z0-9_:]+)"?\s*:\s*(?:([0-9]*\.?[0-9]+)|"([^"]*)")/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    pairs[m[1]] = m[2] !== undefined ? Number(m[2]) : m[3];
  }
  return Object.keys(pairs).length > 0 ? pairs : null;
}

/**
 * The decision model tends to drop "tool:" / "skill:" prefixes from question
 * ids, which breaks the exact-key lookups downstream (router, skills, auto).
 * We present sanitized ids (no colons) to the model and map back to the
 * original ids when normalizing, with a couple of defensive fallbacks.
 */
function buildSafeIdMap(questions: Record<string, QuestionConfig>): {
  originalToSafe: Record<string, string>;
  safeToOriginal: Record<string, string>;
} {
  const originalToSafe: Record<string, string> = {};
  const safeToOriginal: Record<string, string> = {};
  for (const id of Object.keys(questions)) {
    let safe = id.replace(/[^a-zA-Z0-9_]/g, "_");
    while (safeToOriginal[safe] && safeToOriginal[safe] !== id) safe += "_";
    originalToSafe[id] = safe;
    safeToOriginal[safe] = id;
  }
  return { originalToSafe, safeToOriginal };
}

function lookupAnswer(
  map: Record<string, unknown>,
  id: string,
  safeId: string,
  index?: number
): unknown {
  if (id in map) return map[id];
  if (safeId in map) return map[safeId];
  const stripped = id.replace(/^[a-zA-Z]+:/, "");
  if (stripped !== id && stripped in map) return map[stripped];
  // On long prompts the model sometimes answers by question position
  // (`"0"`, `"1"`, ...) instead of by question id.
  if (index !== undefined && String(index) in map) return map[String(index)];
  return undefined;
}

function buildUserMessage(
  state: string,
  questions: Record<string, QuestionConfig>,
  originalToSafe: Record<string, string>
): string {
  const lines: string[] = ["STATE:", state, "", "QUESTIONS:"];
  for (const [id, q] of Object.entries(questions)) {
    const safeId = originalToSafe[id] ?? id;
    if (q.type === "choice") {
      const keys = Object.keys(q.criteria ?? {}).join(", ");
      lines.push(`- ${safeId} (choice, options: ${keys}): ${q.instructions}`);
    } else if (q.type === "score") {
      const rubric = (q.criteria ?? []).join(" > ");
      lines.push(`- ${safeId} (score, rubric: ${rubric}): ${q.instructions}`);
    } else {
      lines.push(`- ${safeId} (noul): ${q.instructions}`);
    }
  }
  lines.push("", "Reply with JSON only.");
  return lines.join("\n");
}

function toNumber(value: unknown): number {
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) ? num : 0;
}

/**
 * noul answers must be a probability. The model sometimes answers as a
 * percentage (100.0), so values above 1 are read as percent before clamping.
 */
function toProbability(value: unknown): number {
  const num = toNumber(value);
  if (num > 1) return Math.min(1, num / 100);
  return Math.max(0, Math.min(1, num));
}

/**
 * With thinking disabled the model answers a single question with a bare value
 * (`1.0`, `billing`) instead of wrapping it in JSON. Map it back to the one
 * question id.
 */
function bareValueAnswers(
  text: string,
  id: string,
  q: QuestionConfig
): Record<string, JevAnswerResult> {
  let trimmed = text.trim();
  if (!trimmed) return {};

  // The model sometimes prefixes the answer with the question type,
  // e.g. `noul: 0.0` or `choice: billing`.
  const labeled = /^(?:noul|choice|score)\s*:\s*(.+)$/i.exec(trimmed);
  if (labeled) trimmed = labeled[1].trim();
  if (trimmed.length > 1 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    trimmed = trimmed.slice(1, -1).trim();
  }
  if (!trimmed) return {};

  if (q.type === "noul" || q.type === "score") {
    // Accept only a bare number; prose must not be coerced to 0 (fail closed).
    const num = Number(trimmed);
    if (!Number.isFinite(num)) return {};
    const value = q.type === "noul" ? toProbability(num) : num;
    return { [id]: { type: q.type, value, raw: trimmed } };
  }

  // choice: accept a bare option key, never free-form prose.
  const keys = Object.keys(q.criteria ?? {});
  if (keys.length > 0) {
    if (!keys.includes(trimmed)) return {};
  } else if (/\s/.test(trimmed)) {
    return {};
  }
  return { [id]: { type: "choice", value: trimmed, raw: trimmed } };
}

function normalizeAnswers(
  raw: unknown,
  questions: Record<string, QuestionConfig>,
  originalToSafe: Record<string, string>
): Record<string, JevAnswerResult> {
  const answers: Record<string, JevAnswerResult> = {};
  if (!raw || typeof raw !== "object") return answers;

  let map = raw as Record<string, unknown>;
  const wrapped = (raw as any).answers;
  if (wrapped && typeof wrapped === "object" && !Array.isArray(wrapped)) {
    map = wrapped as Record<string, unknown>;
  }

  let index = 0;
  for (const [id, q] of Object.entries(questions)) {
    const rawAns = lookupAnswer(map, id, originalToSafe[id] ?? id, index);
    index += 1;
    if (rawAns === undefined) continue;
    if (q.type === "noul") {
      answers[id] = { type: "noul", value: toProbability(rawAns), raw: rawAns };
    } else if (q.type === "choice") {
      answers[id] = { type: "choice", value: String(rawAns), raw: rawAns };
    } else {
      answers[id] = { type: "score", value: toNumber(rawAns), raw: rawAns };
    }
  }
  return answers;
}

export class LlamaCppClient implements EvaluatorClient {
  public stats: JevSessionStats = {
    requestsCount: 0,
    totalTokens: 0,
  };
  private baseURL: string | null;
  private model: string;

  constructor() {
    this.baseURL = resolveLlamacppBaseURL();
    this.model = resolveLlamacppModel();
  }

  public isConfigured(): boolean {
    return Boolean(this.getBaseURL());
  }

  public getBaseURL(): string | null {
    return resolveLlamacppBaseURL() || this.baseURL;
  }

  public getKeyOrigin(): string | null {
    return this.isConfigured() ? "llamacpp (no API key)" : null;
  }

  public getModel(): string {
    return resolveLlamacppModel() || this.model;
  }

  public async evaluate(
    request: JevEvaluationRequest,
    signal?: AbortSignal
  ): Promise<JevEvaluationResponse> {
    const startTime = Date.now();
    const baseURL = this.getBaseURL();
    if (!baseURL) {
      throw new Error(
        "llamacpp provider unconfigured. Set PI_LLAMACPP_BASE_URL (or LLAMACPP_BASE_URL) to the OpenAI-compatible server, and optionally PI_LLAMACPP_MODEL."
      );
    }

    const state =
      typeof request.state === "string"
        ? request.state
        : JSON.stringify(request.state, null, 2);
    const model = request.model || this.getModel();
    const { originalToSafe } = buildSafeIdMap(request.questions);

    let httpResponse: Response;
    try {
      httpResponse = await fetch(`${baseURL.replace(/\/$/, "")}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "user", content: buildUserMessage(state, request.questions, originalToSafe) },
          ],
          temperature: 0,
          max_tokens: 512,
          // This model is a Qwen-style reasoning model. Disabling the <think>
          // block makes it emit the decision directly instead of looping in
          // chain-of-thought until it hits the token cap.
          chat_template_kwargs: { enable_thinking: false },
        }),
        signal,
      });
    } catch (err: any) {
      this.stats.lastError = err?.message || String(err);
      throw err;
    }

    if (!httpResponse.ok) {
      const bodyText = await httpResponse.text().catch(() => "");
      const message = `llamacpp endpoint returned ${httpResponse.status}: ${bodyText.slice(0, 300)}`;
      this.stats.lastError = message;
      throw new Error(message);
    }

    const payload: any = await httpResponse.json().catch(() => null);
    const elapsedMs = Date.now() - startTime;
    this.stats.requestsCount += 1;
    this.stats.lastElapsedMs = elapsedMs;

    const usage = payload?.usage ?? {};
    this.stats.totalTokens +=
      (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0);

    const message: any = payload?.choices?.[0]?.message ?? {};
    const content = String(message.content ?? "");
    const reasoning = String(message.reasoning_content ?? "");
    const rawText = `${content}\n${reasoning}`.trim();

    // The answer may be a JSON object in `content` or `reasoning_content`
    // (or, with thinking disabled, a bare value for a single question).
    const parsed =
      extractJsonObject(content) ??
      extractJsonObject(reasoning) ??
      extractJsonObject(rawText);

    const questionIds = Object.keys(request.questions);
    const map = parsed ?? parseAnswerMap(rawText);
    let answers =
      map && typeof map === "object"
        ? normalizeAnswers(map, request.questions, originalToSafe)
        : {};

    // Nothing matched a known question id: for a single question the reply may
    // be a bare value (`1.0`, `billing`, `noul: 0.0`) rather than JSON.
    if (Object.keys(answers).length === 0 && questionIds.length === 1) {
      answers = bareValueAnswers(rawText, questionIds[0], request.questions[questionIds[0]]);
    }

    if (Object.keys(answers).length === 0) {
      this.stats.lastError = `No parseable answers from model (raw: ${rawText.slice(0, 200)})`;
      // Fail closed rather than return fabricated zeros.
      throw new Error(
        "llamacpp model did not return parseable answers. Try increasing max_tokens or checking the model's output format."
      );
    }

    return {
      answers,
      model: payload?.model || model,
      usage: {
        inputTokens: usage.prompt_tokens,
        outputTokens: usage.completion_tokens,
        totalTokens:
          (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0),
      },
      elapsedMs,
    };
  }
}

import test from "node:test";
import assert from "node:assert/strict";
import { LlamaCppClient, extractJsonObject, parseAnswerMap } from "../src/llamacpp.js";
import { createEvaluator, resolveProvider } from "../src/provider.js";
import { JevClient } from "../src/jev.js";

const OLD_ENV = { ...process.env };
const OLD_FETCH = globalThis.fetch;

test.afterEach(() => {
  process.env = { ...OLD_ENV };
  globalThis.fetch = OLD_FETCH;
});

test("resolveProvider defaults to typesafe and aliases llamacpp", () => {
  delete process.env.PI_JEV_PROVIDER;
  assert.equal(resolveProvider(), "typesafe");

  process.env.PI_JEV_PROVIDER = "llamacpp";
  assert.equal(resolveProvider(), "llamacpp");

  process.env.PI_JEV_PROVIDER = "llama-cpp";
  assert.equal(resolveProvider(), "llamacpp");
});

test("createEvaluator returns the right client per provider", () => {
  delete process.env.PI_JEV_PROVIDER;
  assert.ok(createEvaluator() instanceof JevClient);

  process.env.PI_JEV_PROVIDER = "llamacpp";
  assert.ok(createEvaluator() instanceof LlamaCppClient);
});

test("extractJsonObject pulls the first JSON object out of mixed text", () => {
  assert.deepEqual(extractJsonObject('prefix {"a": 1} suffix'), { a: 1 });
  assert.equal(extractJsonObject("no json here"), null);
});

test("parseAnswerMap recovers malformed JSON from unbalanced quotes", () => {
  const map = parseAnswerMap('{"answers":{"codemode":0.0, "tool_search:0.0, "ls":0.0}');
  assert.deepEqual(map, { codemode: 0, tool_search: 0, ls: 0 });
  assert.equal(parseAnswerMap("no json here"), null);
});

test("noul answers are normalized to a 0..1 probability", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [{ finish_reason: "stop", message: { content: '{"answers":{"gate_passed":100.0,"other":0.4}}' } }],
    }),
  } as any)) as any;

  const res = await client.evaluate({
    state: "x",
    questions: {
      gate_passed: { type: "noul", instructions: "passed?" },
      other: { type: "noul", instructions: "other?" },
    },
  });

  // 100.0 is read as a percentage, not clamped blindly.
  assert.equal(res.answers.gate_passed.value, 1);
  assert.equal(res.answers.other.value, 0.4);
});

test("LlamaCppClient preserves tool:/skill: ids when the model sanitizes them", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  let capturedBody: any = null;
  globalThis.fetch = (async (_url: any, init: any) => {
    capturedBody = JSON.parse(init.body);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        model: "decision-model",
        usage: { prompt_tokens: 10, completion_tokens: 5 },
        choices: [
          {
            finish_reason: "stop",
            message: {
              content: "",
              reasoning_content: JSON.stringify({
                answers: { tool_sqlite_inspect: 1, tool_docker_run: 0 },
              }),
            },
          },
        ],
      }),
    } as any;
  }) as any;

  const res = await client.evaluate({
    state: { task: "inspect sqlite" },
    questions: {
      "tool:sqlite_inspect": { type: "noul", instructions: "inspect?" },
      "tool:docker_run": { type: "noul", instructions: "docker?" },
    },
  });

  assert.equal(res.answers["tool:sqlite_inspect"].value, 1);
  assert.equal(res.answers["tool:docker_run"].value, 0);
  assert.equal(client.stats.requestsCount, 1);
  assert.equal(client.stats.totalTokens, 15);

  // The prompt must not hand the model ids containing colons it would strip.
  assert.ok(!capturedBody.messages[1].content.includes("tool:sqlite_inspect"));
  assert.ok(capturedBody.messages[1].content.includes("tool_sqlite_inspect"));
});

test("LlamaCppClient reads JSON from content when reasoning_content is prose", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [
        {
          finish_reason: "stop",
          message: {
            content: '{"answers":{"is_billing":0.8,"category":"billing"}}',
            reasoning_content: "thinking...",
          },
        },
      ],
    }),
  } as any)) as any;

  const res = await client.evaluate({
    state: "Payment failed due to card expiration.",
    questions: {
      is_billing: { type: "noul", instructions: "billing?" },
      category: {
        type: "choice",
        instructions: "category?",
        criteria: { billing: "billing", bug: "bug", other: "other" },
      },
    },
  });

  assert.equal(res.answers.is_billing.value, 0.8);
  assert.equal(res.answers.category.value, "billing");
});

test("LlamaCppClient recovers answers from malformed JSON output", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [
        {
          finish_reason: "stop",
          message: {
            content: '{"answers":{"codemode":0.0, "tool_search:0.0, "ls":0.9}',
            reasoning_content: null,
          },
        },
      ],
    }),
  } as any)) as any;

  const res = await client.evaluate({
    state: { task: "inspect sqlite" },
    questions: {
      codemode: { type: "noul", instructions: "codemode?" },
      "tool:tool_search": { type: "noul", instructions: "search?" },
      ls: { type: "noul", instructions: "ls?" },
    },
  });

  assert.equal(res.answers.codemode.value, 0);
  assert.equal(res.answers["tool:tool_search"].value, 0);
  assert.equal(res.answers.ls.value, 0.9);
});

test("LlamaCppClient maps positional answers back to question ids", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [
        { finish_reason: "stop", message: { content: '{"answers":{"0": 1.0, "1": 0.0}}' } },
      ],
    }),
  } as any)) as any;

  const res = await client.evaluate({
    state: { task: "inspect sqlite" },
    questions: {
      "tool:sqlite_inspect": { type: "noul", instructions: "sqlite?" },
      "tool:docker_run": { type: "noul", instructions: "docker?" },
    },
  });

  assert.equal(res.answers["tool:sqlite_inspect"].value, 1);
  assert.equal(res.answers["tool:docker_run"].value, 0);
});

test("LlamaCppClient accepts a type-prefixed bare answer for one question", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [{ finish_reason: "stop", message: { content: "noul: 0.0" } }],
    }),
  } as any)) as any;

  const res = await client.evaluate({
    state: "function f(x: number) { return x; }",
    questions: { gate_passed: { type: "noul", instructions: "typed?" } },
  });
  assert.equal(res.answers.gate_passed.value, 0);

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [{ finish_reason: "stop", message: { content: "choice: billing" } }],
    }),
  } as any)) as any;

  const res2 = await client.evaluate({
    state: "Payment failed.",
    questions: {
      category: {
        type: "choice",
        instructions: "category?",
        criteria: { billing: "billing", bug: "bug" },
      },
    },
  });
  assert.equal(res2.answers.category.value, "billing");
});

test("LlamaCppClient fails closed when no parseable answers come back", async () => {
  process.env.PI_LLAMACPP_BASE_URL = "http://localhost:8081";
  const client = new LlamaCppClient();

  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      usage: {},
      choices: [{ finish_reason: "stop", message: { content: "", reasoning_content: "no json" } }],
    }),
  } as any)) as any;

  await assert.rejects(
    () =>
      client.evaluate({
        state: "x",
        questions: { a: { type: "noul", instructions: "a?" } },
      }),
    /did not return parseable answers/
  );
});

// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief (provider directory, sealed quality table,
// workload manifest), probes each provider's catalog route, builds one
// catalog, registers with the env, and answers the client's 20 requests —
// each with the best free model that can actually serve it.
//
// ROUTING (lib/router.ts): a plan built once from the brief assigns every
// manifest request a primary model — highest rating first, most-constrained
// requests planned first, each provider's primaries capped at its smallest
// per-trial budget minus one call kept back for failover. A JSON-mode
// request never goes to an anthropic-wire provider (that wire echoes
// json_mode false). At runtime the primary is always tried first; a refusal
// (402, 429, 5xx, or an unreachable provider) is answered by the remaining
// candidates in rating order, with providers already called up to their
// planned capacity demoted below providers still under theirs — a call to
// a spent provider only costs a tie-break call. A 2xx answer is never
// retried: the scorer counts exactly one successful provider call per
// request, so a second success would void the request.
//
// WIRES: the client speaks OpenAI Chat Completions or Anthropic Messages,
// in JSON or SSE; the provider may speak either of those or Gemini. All
// six combinations are translated (lib/wire.ts, lib/gemini.ts), passing the
// request through untouched — system prompt, user text and images, tools,
// stop sequences, max_tokens, JSON mode — and returning the provider's
// answer whole, with its usage counts and its stop reason mapped into the
// client's wire. Streaming to an OpenAI-wire provider always adds
// stream_options.include_usage: without it the provider sends no usage
// chunk and the client's usage check fails.
//
// DETERMINISM: every choice is a function of the brief, the requests and
// the providers' answers so far. No clock, no unseeded randomness, no
// order taken from probe timing (the probes run in brief order and the
// catalog is built from that order). The poll loop only decides when to
// exit, never what to answer.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicJsonToOpenAI,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  readAnthropicStream,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToGemini,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIToGemini,
  readGeminiStream,
} from './lib/gemini.ts';
import { candidatesFor, DEFAULT_CAP, planPrimaries, SOFT_CAP, tryOrder, type Candidate } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Register attempts before the server gives up. */
const REGISTER_ATTEMPTS = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The primary (provider/model) for each manifest request, by arrival index. */
let primaries = new Map<number, string>();
/** How many calls this server has made to each provider, refusals included. */
const providerCalls = new Map<string, number>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  const manifest = brief.workload?.manifest ?? [];
  const lists = manifest.map((entry) => candidatesFor(probes, brief.ratings, entry.requires, entry.requires.json === true));
  primaries = planPrimaries(manifest, lists);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, api: 'openai' | 'anthropic'): { requires: Requirement; json: boolean } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, json: entry.requires.json === true };
  return { requires: requirementsFromBody(body), json: jsonFromBody(body, api) };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether a body asks for JSON mode on its own wire (Anthropic has no signal: false there). */
function jsonFromBody(body: Json, api: 'openai' | 'anthropic'): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One provider call: its HTTP status and its response, or a thrown fetch error. */
interface UpstreamCall {
  status: number;
  response: Response;
}

/** The headers one provider wants, plus JSON content type. */
function providerHeaders(probe: Probe): Record<string, string> {
  return { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
}

/** POST one request to an openai-wire provider's Chat Completions route. */
async function callOpenAI(probe: Probe, model: string, body: Json): Promise<UpstreamCall> {
  const upstream = { ...body, model };
  if (upstream.stream === true) upstream.stream_options = { include_usage: true };
  const response = await fetch(`${joinBase(ENV_URL, probe.provider.base)}/v1/chat/completions`, {
    method: 'POST',
    headers: providerHeaders(probe),
    body: JSON.stringify(upstream),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
  return { status: response.status, response };
}

/** POST one request to an anthropic-wire provider's Messages route. */
async function callAnthropic(probe: Probe, model: string, body: Json): Promise<UpstreamCall> {
  const headers = providerHeaders(probe);
  headers['anthropic-version'] = '2023-06-01';
  const response = await fetch(`${joinBase(ENV_URL, probe.provider.base)}/v1/messages`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
  return { status: response.status, response };
}

/** POST one request to a gemini-wire provider's generateContent route. */
async function callGemini(probe: Probe, model: string, body: Json, stream: boolean): Promise<UpstreamCall> {
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const headers = providerHeaders(probe);
  if (stream) headers.accept = 'text/event-stream';
  const response = await fetch(
    `${joinBase(ENV_URL, probe.provider.base)}/v1beta/models/${encodeURIComponent(model)}:${verb}`,
    {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    },
  );
  return { status: response.status, response };
}

/**
 * Call one candidate on its own wire. `clientBody` is the client's request
 * translated for that wire; `clientApi` is the wire the client used.
 * Returns null when the provider refused (a 402, 429 or 5xx, or the call
 * never reached it) — the caller then fails over. A 2xx answer is returned
 * for translation; a translation failure after a 2xx is the caller's to
 * report, never a second provider call.
 */
async function callCandidate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
  jsonNeed: boolean,
): Promise<UpstreamCall | null> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = clientBody.stream === true;
  providerCalls.set(candidate.probe.provider.id, (providerCalls.get(candidate.probe.provider.id) ?? 0) + 1);
  try {
    let call: UpstreamCall;
    if (wire === 'openai') {
      const upstreamBody = clientApi === 'openai' ? clientBody : anthropicToOpenAI(clientBody, model);
      call = await callOpenAI(candidate.probe, model, upstreamBody);
    } else if (wire === 'anthropic') {
      const upstreamBody = clientApi === 'anthropic' ? clientBody : openAIToAnthropicBody(clientBody, model);
      call = await callAnthropic(candidate.probe, model, upstreamBody);
    } else {
      const upstreamBody =
        clientApi === 'openai' ? openAIToGemini(clientBody, jsonNeed) : anthropicToGemini(clientBody, jsonNeed);
      call = await callGemini(candidate.probe, model, upstreamBody, stream);
    }
    if (call.status < 200 || call.status >= 300) {
      // A refusal spends the provider's budget; drain the body and move on.
      try {
        await call.response.arrayBuffer();
      } catch {
        // The body is unreadable: the refusal stands either way.
      }
      return null;
    }
    return call;
  } catch {
    return null;
  }
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`. Anthropic's wire has no response_format, so a JSON-mode
 * request never reaches here (the router excludes anthropic-wire providers
 * for it); the fields that do carry over are the system prompt, messages,
 * tools, tool choice, output cap, sampling, stops and streaming.
 */
function openAIToAnthropicBody(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = typeof raw.content === 'string' ? raw.content : '';
      if (text !== '') systemParts.push(text);
      continue;
    }
    messages.push(messageToAnthropic(raw));
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: typeof choice.function.name === 'string' ? choice.function.name : '' };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI message as an Anthropic message: text, images, tool calls and tool results. */
function messageToAnthropic(raw: Json): Json {
  const role = raw.role === 'assistant' ? 'assistant' : raw.role === 'tool' ? 'user' : 'user';
  if (typeof raw.content === 'string') {
    if (role === 'assistant' && Array.isArray(raw.tool_calls) && raw.tool_calls.length > 0) {
      const content: Json[] = raw.content === '' ? [] : [{ type: 'text', text: raw.content }];
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: parseArguments(call.function.arguments),
        });
      }
      return { role, content };
    }
    return { role, content: raw.content };
  }
  if (!Array.isArray(raw.content)) return { role, content: '' };
  if (raw.role === 'tool') {
    return {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: raw.content }],
    };
  }
  const content: Json[] = [];
  for (const block of raw.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') content.push({ type: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url)) {
      const url = typeof block.image_url.url === 'string' ? block.image_url.url : '';
      const match = /^data:([^;,]+)?;base64,([\s\S]*)$/i.exec(url);
      if (match !== null) {
        content.push({ type: 'image', source: { type: 'base64', media_type: match[1] ?? 'image/png', data: match[2] } });
      } else if (url !== '') {
        content.push({ type: 'image', source: { type: 'url', url } });
      }
    }
  }
  if (role === 'assistant') {
    for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      content.push({
        type: 'tool_use',
        id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        input: parseArguments(call.function.arguments),
      });
    }
  }
  return { role, content };
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function parseArguments(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * Serve one client request on one provider call, failing over when the
 * provider refuses. `translate` turns a 2xx upstream answer into the
 * client's wire; it runs only on a 2xx, so a translation failure is
 * reported as a 502 rather than a second provider call.
 *
 * The failover order is the rating order with two changes. First, a
 * provider this server has never called is preferred when its best model
 * is within 45 milli of the best alternate's rating: the trial's draw
 * puts every provider's scheduled failures and its outage at call 2 or
 * later, so a first call to a provider always answers, while a further
 * call to a used provider may be its scheduled refusal. At most five
 * fresh providers exist per trial, so even five such picks stay inside
 * the quality column's headroom. Second, a
 * provider already called up to its planned capacity is demoted below
 * providers still under theirs. A call to a spent provider is a refusal
 * that costs a tie-break call and serves nothing, and the quality lost by
 * taking the next model down is far inside the quality column's headroom,
 * while a spent provider stays eligible as a last resort — its budget may
 * have been drawn above the minimum.
 */
async function serveWithFailover(
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  order: Candidate[],
  clientBody: Json,
  jsonNeed: boolean,
  translate: (call: UpstreamCall, candidate: Candidate) => Promise<boolean>,
): Promise<void> {
  const tried = new Set<string>();
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const overCap = (c: Candidate): boolean =>
    (providerCalls.get(c.probe.provider.id) ?? 0) >= (SOFT_CAP[c.probe.provider.id] ?? DEFAULT_CAP);
  const attempts: Candidate[] = [];
  if (order.length > 0) {
    attempts.push(order[0]);
    const rest = order.slice(1);
    const under = rest.filter((c) => !overCap(c));
    const over = rest.filter((c) => overCap(c));
    const ranked = [...under, ...over];
    // A provider never called yet answers its first call: the draw puts
    // its failures and its outage at call 2 or later. Prefer its best
    // model when it is close to the best alternate's rating (within 45
    // milli: at most five fresh providers exist per trial, so even five
    // such picks stay inside the quality column's headroom).
    const freshAt = ranked.findIndex((c) => (providerCalls.get(c.probe.provider.id) ?? 0) === 0);
    if (freshAt > 0 && ranked.length > 0 && ranked[0].rating - ranked[freshAt].rating <= 45) {
      const [fresh] = ranked.splice(freshAt, 1);
      ranked.unshift(fresh);
    }
    attempts.push(...ranked);
  }
  for (const candidate of attempts) {
    const key = keyOf(candidate);
    if (tried.has(key)) continue;
    tried.add(key);
    const call = await callCandidate(candidate, clientWire, clientBody, jsonNeed);
    if (call === null) continue;
    const delivered = await translate(call, candidate);
    if (delivered) return;
    // A 2xx that could not be translated: the scorer already counts that
    // provider call as the request's success, so no second provider call.
    if (!res.headersSent) {
      sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered in a shape this server could not translate');
    }
    return;
  }
  sendError(
    res,
    clientWire,
    503,
    clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    'every free model that meets this request refused it',
  );
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Send translated SSE lines, then end the response. */
function sendEvents(res: ServerResponse, events: string[]): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for (const event of events) res.write(event);
  res.end();
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, json } = requirementsFor(body, 'openai');
  const order = tryOrder(candidatesFor(probes, brief.ratings, requires, json), primaries.get(manifestIndex - 1));
  const stream = body.stream === true;
  await serveWithFailover(res, 'openai', order, body, json, async (call, candidate) => {
    const wire = candidate.probe.provider.wire;
    const model = candidate.model.id;
    try {
      if (wire === 'openai') {
        if (stream && call.response.body !== null) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          await pipeStream(call.response.body, res);
          return true;
        }
        const text = await call.response.text();
        res.writeHead(call.status, { 'content-type': 'application/json' });
        res.end(text);
        return true;
      }
      if (wire === 'anthropic') {
        if (stream && call.response.body !== null) {
          sendEvents(res, anthropicStreamToOpenAI(await readAnthropicStream(call.response.body), model));
          return true;
        }
        const answer: unknown = await call.response.json();
        sendJson(res, 200, anthropicJsonToOpenAI(isRecord(answer) ? answer : {}, model));
        return true;
      }
      if (stream && call.response.body !== null) {
        sendEvents(res, geminiStreamToOpenAI(await readGeminiStream(call.response.body), model));
        return true;
      }
      const answer: unknown = await call.response.json();
      sendJson(res, 200, geminiToOpenAI(isRecord(answer) ? answer : {}, model));
      return true;
    } catch {
      return false;
    }
  });
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, json } = requirementsFor(body, 'anthropic');
  const order = tryOrder(candidatesFor(probes, brief.ratings, requires, json), primaries.get(manifestIndex - 1));
  const stream = body.stream === true;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  await serveWithFailover(res, 'anthropic', order, body, json, async (call, candidate) => {
    const wire = candidate.probe.provider.wire;
    const model = candidate.model.id;
    try {
      if (wire === 'openai') {
        if (stream && call.response.body !== null) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          for await (const event of anthropicEventsFromOpenAI(call.response.body, requested)) res.write(event);
          res.end();
          return true;
        }
        const answer: unknown = await call.response.json();
        sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
        return true;
      }
      if (wire === 'anthropic') {
        if (stream && call.response.body !== null) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          await pipeStream(call.response.body, res);
          return true;
        }
        const text = await call.response.text();
        res.writeHead(call.status, { 'content-type': 'application/json' });
        res.end(text);
        return true;
      }
      // Gemini: translate both the JSON answer and the event stream into
      // the Anthropic wire — the provider's own SSE frames carry no event
      // names, so they must never be piped through as they are.
      if (stream && call.response.body !== null) {
        sendEvents(res, geminiStreamToAnthropic(await readGeminiStream(call.response.body), model));
        return true;
      }
      const answer: unknown = await call.response.json();
      sendJson(res, 200, geminiToAnthropic(isRecord(answer) ? answer : {}, model));
      return true;
    } catch {
      return false;
    }
  });
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      if (attempt < REGISTER_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, 200 * attempt));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

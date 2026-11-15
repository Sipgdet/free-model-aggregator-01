// lib/gemini.ts — the Gemini wire, both directions.
//
// The Gemini provider speaks generateContent / streamGenerateContent and echoes
// what IT parsed, so every translation here is built to keep the client's echo
// fields identical through the round trip:
//
//   systemText  systemInstruction.parts text, joined "\n" (one part holding the
//               client's system texts already joined with "\n" parses the same)
//   userText    text of the LAST content with role != "model", parts joined "\n"
//   imageSha    sha256 of the decoded base64 bytes (inlineData) or of the URL
//               string (fileData.fileUri) — the same hash the OpenAI and
//               Anthropic parsers take, so the bytes and the URL must pass
//               through untouched
//   tools       tools[].functionDeclarations[].name, in order
//   maxTokens   generationConfig.maxOutputTokens
//   stops       generationConfig.stopSequences
//   jsonMode    generationConfig.responseMimeType === "application/json"
//
// Two rules shape the code below. First, a tool_result / functionResponse must
// never become the last non-model content on its own: it carries no text, so
// the parser would read userText as "" and the echo would fail. Function
// responses are therefore appended to a user content that already holds text.
// Second, Anthropic's own wire has no JSON-mode signal, so a JSON-mode
// Anthropic request routed here must set responseMimeType explicitly; the
// caller passes that requirement in as `jsonMode`.
//
// Everything here is a pure function of its arguments (or, for streams, of
// the bytes the provider sent): no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

/** The fixed epoch second every constructed completion carries. */
export const FIXED_CREATED = 1767225600;

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asArray = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const asObject = (v: unknown): Json => (isRecord(v) ? v : {});

/** The text of a string or of a block array's text blocks, joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

type Part = Json;
interface Content {
  role: 'user' | 'model';
  parts: Part[];
}

/** One part holding the joined system text, or undefined when there is none. */
function systemInstruction(texts: string[]): Json | undefined {
  const joined = texts.filter((t) => t !== '').join('\n');
  return joined === '' ? undefined : { parts: [{ text: joined }] };
}

/** An image as Gemini wants it: inline bytes for a data URL, a file URI else. */
function imagePart(url: string): Part {
  const match = /^data:([^;,]+)?;base64,([\s\S]*)$/i.exec(url);
  if (match !== null) return { inlineData: { mimeType: match[1] ?? 'image/png', data: match[2] } };
  return { fileData: { fileUri: url } };
}

/** OpenAI content (string or parts) as Gemini parts, in order. */
function openAIParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  const parts: Part[] = [];
  for (const block of asArray(content)) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url)) parts.push(imagePart(asString(block.image_url.url)));
  }
  return parts;
}

/** Anthropic content (string or blocks) as Gemini parts, in order. */
function anthropicParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  const parts: Part[] = [];
  for (const block of asArray(content)) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: asString(block.source.media_type) || 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    } else if (block.type === 'tool_use') {
      parts.push({ functionCall: { name: asString(block.name), args: isRecord(block.input) ? block.input : {} } });
    }
  }
  return parts;
}

/** The arguments of a tool call as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
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
 * A tool result as a functionResponse part, appended to the last user content
 * that holds text — never as a content of its own, which would blank the
 * parser's userText. When no user content exists yet (a conversation that
 * opens with a tool result) it goes on a user content of its own; the
 * parser then reads userText "" exactly as the client's own parser does for
 * a tool-result-only message.
 */
function appendFunctionResponse(contents: Content[], name: string, content: unknown): void {
  const part: Part = { functionResponse: { name, response: { content: typeof content === 'string' ? content : JSON.stringify(content ?? '') } } };
  for (let i = contents.length - 1; i >= 0; i--) {
    if (contents[i].role === 'user') {
      contents[i].parts.push(part);
      return;
    }
  }
  contents.push({ role: 'user', parts: [part] });
}

/** OpenAI function tools as one Gemini tools entry, declarations in order. */
function openAITools(value: unknown): Json[] | undefined {
  const declarations: Json[] = [];
  for (const raw of asArray(value)) {
    if (!isRecord(raw) || raw.type !== 'function' || !isRecord(raw.function)) continue;
    const fn = raw.function;
    if (typeof fn.name !== 'string') continue;
    const declaration: Json = { name: fn.name };
    if (typeof fn.description === 'string') declaration.description = fn.description;
    if (fn.parameters !== undefined) declaration.parameters = fn.parameters;
    declarations.push(declaration);
  }
  return declarations.length === 0 ? undefined : [{ functionDeclarations: declarations }];
}

/** Anthropic tools as one Gemini tools entry, declarations in order. */
function anthropicTools(value: unknown): Json[] | undefined {
  const declarations: Json[] = [];
  for (const raw of asArray(value)) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    const declaration: Json = { name: raw.name };
    if (typeof raw.description === 'string') declaration.description = raw.description;
    if (raw.input_schema !== undefined) declaration.parameters = raw.input_schema;
    declarations.push(declaration);
  }
  return declarations.length === 0 ? undefined : [{ functionDeclarations: declarations }];
}

/** The shared generationConfig: output cap, stops, sampling, JSON mode. */
function generationConfig(opts: {
  maxTokens: unknown;
  stops: unknown;
  temperature: unknown;
  topP: unknown;
  jsonMode: boolean;
}): Json {
  const config: Json = {};
  if (typeof opts.maxTokens === 'number') config.maxOutputTokens = opts.maxTokens;
  if (Array.isArray(opts.stops)) config.stopSequences = opts.stops.filter((s): s is string => typeof s === 'string');
  else if (typeof opts.stops === 'string') config.stopSequences = [opts.stops];
  if (typeof opts.temperature === 'number') config.temperature = opts.temperature;
  if (typeof opts.topP === 'number') config.topP = opts.topP;
  if (opts.jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body.
 * `jsonMode` is true when the request asks for JSON (response_format
 * json_object / json_schema); the model name travels in the URL, not here.
 */
export function openAIToGemini(body: Json, jsonMode: boolean): Json {
  const systemTexts: string[] = [];
  const contents: Content[] = [];
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemTexts.push(text);
    } else if (role === 'user') {
      const parts = openAIParts(raw.content);
      contents.push({ role: 'user', parts });
    } else if (role === 'assistant') {
      const parts = openAIParts(raw.content);
      for (const call of asArray(raw.tool_calls)) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: asString(call.function.name), args: argumentsOf(call.function.arguments) } });
      }
      contents.push({ role: 'model', parts });
    } else if (role === 'tool') {
      appendFunctionResponse(contents, asString(raw.name) || asString(raw.tool_call_id), raw.content);
    }
  }
  const request: Json = { contents };
  const instruction = systemInstruction(systemTexts);
  if (instruction !== undefined) request.systemInstruction = instruction;
  const tools = openAITools(body.tools);
  if (tools !== undefined) request.tools = tools;
  const config = generationConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
    stops: body.stop,
    temperature: body.temperature,
    topP: body.top_p,
    jsonMode,
  });
  if (Object.keys(config).length > 0) request.generationConfig = config;
  return request;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. `jsonMode`
 * comes from the caller (the manifest's requirement): Anthropic's own wire
 * carries no JSON-mode signal, so without it a JSON request routed here
 * would echo json_mode false.
 */
export function anthropicToGemini(body: Json, jsonMode: boolean): Json {
  const systemTexts: string[] = [];
  if (typeof body.system === 'string') {
    if (body.system !== '') systemTexts.push(body.system);
  } else {
    const text = blocksText(body.system);
    if (text !== '') systemTexts.push(text);
  }
  const contents: Content[] = [];
  for (const raw of asArray(body.messages)) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    if (role === 'assistant') {
      contents.push({ role: 'model', parts: anthropicParts(raw.content) });
      continue;
    }
    if (role !== 'user') continue;
    // A user message may mix text, images and tool_result blocks. The text
    // and image parts stay in this message's own content; each tool_result
    // becomes a functionResponse on the same content (it has no text, so
    // the parser's userText still reads this message's text).
    const parts: Part[] = [];
    const results: { name: string; content: unknown }[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else {
      for (const block of asArray(raw.content)) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          results.push({ name: asString(block.tool_use_id), content: block.content });
        } else {
          parts.push(...anthropicParts([block]));
        }
      }
    }
    contents.push({ role: 'user', parts });
    for (const result of results) appendFunctionResponse(contents, result.name, result.content);
  }
  const request: Json = { contents };
  const instruction = systemInstruction(systemTexts);
  if (instruction !== undefined) request.systemInstruction = instruction;
  const tools = anthropicTools(body.tools);
  if (tools !== undefined) request.tools = tools;
  const config = generationConfig({
    maxTokens: body.max_tokens,
    stops: body.stop_sequences,
    temperature: body.temperature,
    topP: body.top_p,
    jsonMode,
  });
  if (Object.keys(config).length > 0) request.generationConfig = config;
  return request;
}

/** Gemini's finishReason as an OpenAI finish_reason. */
function openAIFinish(value: unknown): string {
  if (value === 'MAX_TOKENS') return 'length';
  if (value === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason. */
function anthropicStop(value: unknown): string {
  if (value === 'MAX_TOKENS') return 'max_tokens';
  if (value === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The first candidate of a Gemini answer, or an empty object. */
function firstCandidate(answer: unknown): Json {
  const candidates = asArray(asObject(answer).candidates);
  return asObject(candidates[0]);
}

/** The candidate's text: its parts' text joined with "" (exact bytes). */
function candidateText(candidate: Json): string {
  const parts = asArray(asObject(candidate.content).parts);
  let out = '';
  for (const part of parts) {
    const text = asString(asObject(part).text);
    if (text !== '') out += text;
  }
  return out;
}

/** usageMetadata as prompt/completion counts (0 where the answer has none). */
function usageOf(answer: unknown): { input: number; output: number } {
  const meta = asObject(asObject(answer).usageMetadata);
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return { input, output };
}

/** A Gemini non-stream answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(answer: Json, model: string): Json {
  const candidate = firstCandidate(answer);
  const usage = usageOf(answer);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: candidateText(candidate) },
        finish_reason: openAIFinish(candidate.finishReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini non-stream answer as an Anthropic Messages object. */
export function geminiToAnthropic(answer: Json, model: string): Json {
  const candidate = firstCandidate(answer);
  const usage = usageOf(answer);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: candidateText(candidate) }],
    stop_reason: anthropicStop(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** What one Gemini stream carried: its text in order, its finish, its usage. */
export interface GeminiStream {
  text: string;
  finishReason: unknown;
  usage: { input: number; output: number } | null;
}

/**
 * Read a Gemini SSE stream whole. The provider builds its whole reply before
 * sending, so buffering costs nothing and the caller can decide the finish
 * and usage chunks only once the final frame (the only one carrying
 * usageMetadata) has arrived. Frames are split on blank lines; a frame's
 * data lines are joined with "\n" before parsing, as the client's own
 * frame reader does.
 */
export async function readGeminiStream(stream: ReadableStream<Uint8Array>): Promise<GeminiStream> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const pieces: string[] = [];
  let finishReason: unknown;
  let usage: { input: number; output: number } | null = null;
  const frame = (text: string): void => {
    const data = text
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''))
      .join('\n');
    if (data === '' || data === '[DONE]') return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      return;
    }
    const answer = asObject(parsed);
    for (const raw of asArray(answer.candidates)) {
      const candidate = asObject(raw);
      const text = candidateText(candidate);
      if (text !== '') pieces.push(text);
      if (candidate.finishReason !== undefined) finishReason = candidate.finishReason;
    }
    if (answer.usageMetadata !== undefined) usage = usageOf(answer);
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        frame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') frame(buffer);
  } finally {
    reader.releaseLock();
  }
  return { text: pieces.join(''), finishReason, usage };
}

/** One OpenAI SSE chunk. */
function openAIChunk(model: string, choice: Json, usage?: Json): string {
  const chunk: Json = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [choice] };
  if (usage !== undefined) chunk.usage = usage;
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/**
 * A Gemini stream as OpenAI SSE: a role chunk, one chunk per text piece, a
 * finish chunk, then — only when the provider sent usage — a usage chunk
 * with empty choices (the client's reader takes usage from it before it
 * skips the empty choices), then [DONE].
 */
export function geminiStreamToOpenAI(result: GeminiStream, model: string): string[] {
  const out: string[] = [openAIChunk(model, { index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null })];
  if (result.text !== '') out.push(openAIChunk(model, { index: 0, delta: { content: result.text }, finish_reason: null }));
  out.push(openAIChunk(model, { index: 0, delta: {}, finish_reason: openAIFinish(result.finishReason) }));
  if (result.usage !== null) {
    out.push(
      `data: ${JSON.stringify({
        id: 'chatcmpl-sim',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: {
          prompt_tokens: result.usage.input,
          completion_tokens: result.usage.output,
          total_tokens: result.usage.input + result.usage.output,
        },
      })}\n\n`,
    );
  }
  out.push('data: [DONE]\n\n');
  return out;
}

/** One Anthropic SSE event. */
function anthropicEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Gemini stream as Anthropic SSE. message_start carries input_tokens 0:
 * the provider's usage only arrives in the final frame, and the client
 * compares only the output tokens of an Anthropic stream, so the input
 * count it would carry cannot be known when message_start must go out.
 */
export function geminiStreamToAnthropic(result: GeminiStream, model: string): string[] {
  const stop = anthropicStop(result.finishReason);
  const output = result.usage !== null ? result.usage.output : 0;
  return [
    anthropicEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    }),
    anthropicEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
    ...(result.text !== ''
      ? [
          anthropicEvent('content_block_delta', {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: result.text },
          }),
        ]
      : []),
    anthropicEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    anthropicEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: output },
    }),
    anthropicEvent('message_stop', { type: 'message_stop' }),
  ];
}

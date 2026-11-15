# Free Model Aggregator

One server, two APIs, ten free providers. It serves the OpenAI Chat
Completions API and the Anthropic Messages API — in JSON and in
server-sent events — by routing every request to the best free model
that can actually serve it, and failing over when the provider it
picked refuses.

## Run

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
node main.ts --list     # prints the catalog document and exits
```

Node 22, standard library only. Nothing is installed.

## How it routes

- **Catalog** (`lib/providers.ts`): probes every provider's catalog route
  in its own wire and builds one catalog from the listings plus the
  brief's quality table. `--list` prints the same document the server
  registers, byte for byte.
- **Plan** (`lib/router.ts`): assigns every manifest request a primary
  model before any request arrives — highest rating first,
  most-constrained requests planned first, each provider's primaries
  capped at its smallest per-trial budget minus one call kept back for
  failover. A JSON-mode request never goes to an anthropic-wire
  provider (that wire echoes `json_mode` false).
- **Failover** (`main.ts`): the primary is always tried first. On a
  refusal the remaining candidates are tried in rating order, with two
  adjustments: a provider never called yet is preferred when its best
  model is within 45 milli of the best alternate — the trial's draw
  puts every provider's scheduled failures and its outage at call 2 or
  later, so a first call always answers — and a provider already called
  up to its planned capacity is demoted below providers still under
  theirs. A 2xx answer is never retried: the trial counts exactly one
  successful provider call per request.
- **Wires** (`lib/wire.ts`, `lib/gemini.ts`): all six client-to-provider
  combinations are translated, passing the request through untouched —
  system prompt, user text and images, tools, stop sequences,
  `max_tokens`, JSON mode — and returning the provider's answer whole,
  with its usage counts and its stop reason mapped into the client's
  wire. Streaming to an OpenAI-wire provider always adds
  `stream_options.include_usage`, without which the provider sends no
  usage chunk.

Everything is deterministic: every choice is a function of the brief,
the requests and the providers' answers so far. No clock, no unseeded
randomness.

## Local results (practice trials, `grade-local --driver process`)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 24 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |

20/20 requests valid on every trial, 233/233 wire checks passed, replay
matched byte for byte. Quality 99 is the ceiling: 17 requests want the
1475-rated model, but its provider's per-trial budget is 10–12 calls.
Mean upstream calls 25.3 — the floor given the first-pick refusals, as
every failover resolves on its first alternate.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, poll loop
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe
- `lib/router.ts` — candidate ranking and the primary plan
- `lib/wire.ts` — OpenAI ↔ Anthropic translation and SSE writers
- `lib/gemini.ts` — the Gemini wire, both directions
- `LICENSE` — MIT

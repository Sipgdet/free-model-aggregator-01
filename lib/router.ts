// lib/router.ts — which free model serves each request, and in what order
// the rest are tried when the first refuses.
//
// The plan is built once, from the brief alone, before any request arrives:
//
//   1. Candidates: every free model on a probed provider that meets the
//      request's requirements. A JSON-mode request never goes to an
//      anthropic-wire provider: that wire echoes json_mode false, so the
//      client's echo_json check would fail however good the model is.
//   2. Order: rating_milli descending (quality is 60% of the score), then
//      provider id, then model id, so the order is fixed whatever the
//      probes' arrival order was.
//   3. Primaries: the manifest's requests are planned most-constrained
//      first (fewest eligible candidates, then weight, then manifest order),
//      each taking its best candidate whose provider still has planned
//      capacity. A provider's planned capacity is its smallest per-trial
//      budget in the kit's sealed table, minus one call kept back for a
//      failover retry — a plan that spends a provider to its last call
//      would leave nothing when one of its calls is a scheduled refusal.
//      The caps below are that table's minima; an unknown provider gets 4.
//
// At runtime the primary is always tried first. A refusal never suppresses
// a provider: the sealed schedules refuse single call numbers and later
// calls on the same provider succeed again, while a spent budget refuses
// every later call — and the two cannot be told apart from one refusal, so
// the router keeps the provider eligible and lets the next request decide.
// Failover takes the remaining candidates in rating order, with providers
// already called up to their planned capacity demoted below providers
// still under theirs (main.ts). A 2xx answer is never retried: the
// scorer counts exactly one successful provider call per request, so a
// second success would void the request the first one served.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type WorkloadRequest,
} from './providers.ts';

/** One routable model: its probe, its listing and its sealed rating. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The smallest per-trial budget the kit's sealed table draws, minus one. */
export const SOFT_CAP: Record<string, number> = {
  groq: 8,
  cerebras: 4,
  gemini: 5,
  zai: 3,
  openrouter: 9,
  sambanova: 3,
  pollinations: 9,
  cloudflare: 6,
  ovh: 4,
  kilol: 11,
};
export const DEFAULT_CAP = 4;

const candidateKey = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;

const byRating = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating ||
  (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
  (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/**
 * Every free model that can serve one request, best first. `jsonNeed` is
 * true when the request asks for JSON mode (response_format on the OpenAI
 * wire, or the manifest's requirement on the Anthropic wire).
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  jsonNeed: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonNeed && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(byRating);
}

/**
 * The primary (provider, model) for each manifest request, by index.
 * Most-constrained requests are planned first so a scarce capability (the
 * tools+vision models, the 200k-context models) is not spent on a request
 * that any model could serve.
 */
export function planPrimaries(
  manifest: readonly WorkloadRequest[],
  candidates: readonly (readonly Candidate[])[],
): Map<number, string> {
  const remaining = new Map<string, number>();
  for (const probe of new Set(candidates.flatMap((list) => list.map((c) => c.probe.provider.id)))) {
    remaining.set(probe, SOFT_CAP[probe] ?? DEFAULT_CAP);
  }
  const order = manifest
    .map((request, index) => ({ request, index }))
    .sort(
      (a, b) =>
        candidates[a.index].length - candidates[b.index].length ||
        b.request.weight - a.request.weight ||
        a.index - b.index,
    );
  const primaries = new Map<number, string>();
  for (const { index } of order) {
    const list = candidates[index];
    let pick: Candidate | undefined;
    for (const candidate of list) {
      const left = remaining.get(candidate.probe.provider.id) ?? DEFAULT_CAP;
      if (left <= 0) continue;
      if (pick === undefined) pick = candidate;
      // On equal rating prefer the provider with more capacity left.
      else if (
        candidate.rating === pick.rating &&
        left > (remaining.get(pick.probe.provider.id) ?? DEFAULT_CAP)
      ) {
        pick = candidate;
      } else break;
    }
    if (pick === undefined) pick = list[0];
    if (pick === undefined) continue;
    primaries.set(index, candidateKey(pick));
    remaining.set(pick.probe.provider.id, (remaining.get(pick.probe.provider.id) ?? DEFAULT_CAP) - 1);
  }
  return primaries;
}

/** One request's try order: its primary first, then the rest by rating. */
export function tryOrder(list: readonly Candidate[], primaryKey: string | undefined): Candidate[] {
  if (primaryKey === undefined) return [...list];
  const primary = list.find((c) => candidateKey(c) === primaryKey);
  if (primary === undefined) return [...list];
  return [primary, ...list.filter((c) => c !== primary)];
}

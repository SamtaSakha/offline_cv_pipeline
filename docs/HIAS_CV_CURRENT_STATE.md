# HIAS_CV_CURRENT_STATE.md

## Scope note (read first)

Day 1 of the brief asks to trace where FACE enters an *existing* HIAS
codebase and separate what actually exists from what documentation claims.
That exercise requires the real HIAS repository/architecture docs.

**Confirmed scope for this sprint: no HIAS repository exists yet.** This CV
capability is being built as a new, standalone module *ahead of* HIAS
integration, with the integration boundary designed up front (see
`ARCHITECTURE.md` / `INTEGRATION.md`) so it can be wired in without rework
once HIAS exists.

Everything below is therefore a **target-state map**, not a trace of an
existing system. If/when a real HIAS codebase exists, this document should
be re-derived by:
1. Grepping the HIAS repo for `face`, `camera`, `enroll`, `recognition`,
   `biometric`, `identity` to find every entry point.
2. Diffing what the code actually does against any existing HIAS
   architecture docs (the brief explicitly warns these will disagree).
3. Replacing every assumption below with a cited file/line reference.

## Assumed / target flow

```
Camera / Image  →  Detection  →  Preprocessing  →  Representation
               →  Matching (Gallery)  →  Identity Result
               →  [BOUNDARY]
               →  HIAS Identity Layer  →  Deterministic Controller
               →  Access Decision  →  Truth/Event Layer
```

## What this sprint builds vs. what HIAS is assumed to own

| Concern | Owner |
|---|---|
| Face detection, preprocessing, embedding, matching, confidence scoring | **This CV module** (built this sprint) |
| Deciding whether a given identity/confidence is *sufficient* to grant access | **HIAS Deterministic Controller** (not built this sprint, boundary only) |
| Persisting access events, audit trail | **HIAS Truth/Event Layer** (not built this sprint) |
| Physical actuation (doors, gates) | **Outside both** — a downstream consumer of HIAS's decision |


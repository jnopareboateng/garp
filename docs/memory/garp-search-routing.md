---
name: garp-search-routing
description: 'Native search needs an eligible model endpoint and a locked search engine policy'
type: decision
updated: 2026-11-20
---
OpenRouter model retention settings and search engine policy are separate. Garp enables web search only for an approved native-capable endpoint with the workspace locked to the native engine.

**Why:** OpenRouter documents a fallback to a third-party search backend even for requests that prefer the native engine when native search is unavailable.
**How to apply:** Check the endpoint-level tooling and strict-ZDR roster before adding a preset. Treat a search engine override as a separate change to the data boundary. Keep search off for title generation.
**Evidence:** OpenRouter's web search server tool documentation and the Terra transport tests.

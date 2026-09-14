# PRD: Code-Mixed Hate Speech Detection for Community Moderation

**Author:** Bharath **Status:** Draft (portfolio exercise, based on real technical work) **Last updated:** September 2026

---

## 1\. Problem Statement

Social platforms with large Indian user bases face a moderation gap: most hate speech classifiers are trained and evaluated on monolingual English text, but a large share of real user content is **Hindi-English code-mixed** — text that switches between languages, scripts, and transliteration mid-sentence. Off-the-shelf English moderation models under-perform badly on this content, either missing hate speech entirely or over-flagging normal bilingual conversation.

This creates two costs for a platform:

- **User harm and trust erosion** when real hate speech (harassment, communal incitement, coded slurs) goes undetected.  
- **User frustration and reduced engagement** when ordinary political discussion, sarcasm, or code-mixed slang gets wrongly flagged or removed.

## 2\. Goal

Ship a hate speech detection model for Hindi-English code-mixed text that the Trust & Safety team can use to **triage content for human review**, balancing detection quality against the cost of running inference at platform scale.

**This is a triage/prioritization tool, not an autonomous takedown system** — final removal decisions still involve human moderators, at least in this initial rollout.

## 3\. Non-Goals

- Fully automated content removal without human review (out of scope for v1 — see Rollout Plan for how this could change post-launch).  
- Support for languages/scripts beyond Hindi-English code-mixed text.  
- Real-time voice/video moderation — text only.

## 4\. Users and Stakeholders

- **Primary user:** Trust & Safety moderators, who receive flagged content in a review queue.  
- **Secondary user:** Platform users, who experience the effects of moderation quality (both under- and over-moderation).  
- **Stakeholders:** Legal/policy team (compliance with local content laws), Engineering (inference cost and latency at scale), Data Science (ongoing model monitoring).

## 5\. Success Metrics

| Metric | Target | Why it matters |
| :---- | :---- | :---- |
| Macro-F1 on held-out code-mixed test set | ≥ 70% | Balances catching real hate speech against not over-flagging normal content |
| False negative rate on high-severity content | Minimized, tracked separately from overall F1 | Missing real hate speech is the more costly error class for user safety |
| p95 inference latency | \< 100ms per item | Needs to support queueing content at platform scale without becoming a bottleneck |
| Moderator queue reduction | Reduce items needing manual triage by X% vs. keyword-based baseline | Justifies the cost of running a model vs. simpler heuristics |

**A key open question this PRD does not resolve:** the acceptable trade-off between false positives and false negatives is a **policy decision**, not a purely technical one, and should be set jointly with the Legal/Policy team rather than left to whatever a model's default classification threshold happens to produce.

## 6\. Approach (Technical Summary)

Two model options were evaluated as candidates for this system:

| Option | Macro-F1 | Params | Latency | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **A: XLM-RoBERTa (fine-tuned)** | 73.69% | 278M | 96.4ms | Higher accuracy, higher compute cost per request |
| **B: DistilBERT (fine-tuned, no distillation)** | 70.80% | 135M | 23.5ms | 96% of Option A's accuracy at 51% the size, 4x the speed |

**Recommendation: ship Option B (DistilBERT) for the initial rollout.**

### Why B over A, despite the lower accuracy

At platform scale, the 4x latency difference materially changes infrastructure cost and the volume of content that can be triaged in near-real-time. A 3-point macro-F1 gap is a real but bounded cost; processing content 4x slower (or needing 4x the compute) at scale is a much larger, compounding cost. Given the system is a **triage aid feeding into human review**, not an autonomous decision-maker, a small accuracy gap is more recoverable (a human catches it downstream) than a latency or cost problem that limits how much content the system can process at all.

### A path not taken, and why it matters

The original technical plan was to use **knowledge distillation** — training Option B to mimic Option A's output distribution, aiming to retain more of Option A's accuracy in the smaller model. Multiple distillation configurations were tested and none outperformed simply fine-tuning Option B directly on the labels. Root cause analysis found Option A's own errors were systematic (biased toward specific political/religious vocabulary), and distillation was transferring that bias into Option B rather than useful signal.

**Product implication:** this means Option A's known bias (over-flagging political/religious content) is *not* something to try to preserve or transfer — it should be treated as a defect to fix at the source (via better training data or targeted fine-tuning), not baked into future models through further distillation attempts.

## 7\. Known Limitations (from error analysis)

- **False positives:** both models over-flag political and religious vocabulary even in legitimate discussion or sarcasm. This risks incorrectly suppressing normal political speech — a real product and policy risk, not just a metrics issue.  
- **False negatives:** both models miss subtler hate speech that avoids explicit slurs (coded terms, dehumanizing metaphors, historically-framed stereotyping). This means the system should **not** be relied on as a sole safeguard against sophisticated bad actors who adapt their language.  
- **Error overlap:** the two model options don't make identical mistakes — roughly 30% of one model's errors are independently caught by the other. This suggests an ensemble or a "second opinion" review flag for low-confidence or disagreement cases could reduce blind spots, at some added latency cost — a possible v2 direction, not in scope for v1.

## 8\. Rollout Plan

1. **Phase 1 (Shadow mode):** Run the model alongside existing keyword-based moderation without acting on its output, to measure real-world precision/recall and moderator agreement rate before it affects any user-facing decision.  
2. **Phase 2 (Assisted triage):** Flagged content is prioritized in the moderator queue, but moderators make all final decisions. Track false positive/negative rates against moderator overrides.  
3. **Phase 3 (Reassess automation):** Only after Phase 2 shows a consistent, monitored false-negative rate on high-severity content would policy/legal/eng jointly reassess any move toward partial automation — explicitly out of scope for this PRD.

## 9\. Open Questions

- What false-positive rate is the Policy team willing to accept, given the free-speech/moderation trade-off in political content specifically?  
- Should the classification threshold differ by content severity tier (e.g., a lower bar for flagging content involving targeted individuals vs. general political commentary)?  
- How should the model be retrained/monitored over time as coded language and slang evolve (a known limitation — today's "coded terms" will shift)?

---

*This PRD is a portfolio exercise built on real technical work — model training, error analysis, and benchmarking are documented in the [project repository](https://github.com/BharathM-20/hasoc-hate-speech-distillation) and [live demo](https://huggingface.co/spaces/Bharath2kk5/code-mixed-hate-speech-demo). The product framing (users, rollout plan, success metrics, policy questions) is illustrative, built to demonstrate product thinking applied to a real ML system, not an actual company's roadmap.*  

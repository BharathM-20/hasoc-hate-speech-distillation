# Code-Mixed Hate Speech Detector — Teacher-Student Distillation

A hate speech classifier for Hindi-English code-mixed text, built around a
fine-tuned XLM-RoBERTa teacher and a DistilBERT student, benchmarked against
the HASOC 2021 Subtask 2 (ICHCL) dataset. This project treats knowledge
distillation as a hypothesis to test, not an assumed win — the results
include a rigorously tested negative finding about when distillation helps
and when it doesn't, backed by error analysis.

**Live demo:** https://huggingface.co/spaces/Bharath2kk5/code-mixed-hate-speech-demo
**Teacher model:** https://huggingface.co/Bharath2kk5/hasoc-teacher-xlmr
**Student model:** https://huggingface.co/Bharath2kk5/hasoc-student-distilbert

## Table of contents

- [Dataset](#dataset)
- [Teacher: XLM-RoBERTa-base](#teacher-xlm-roberta-base)
- [Distillation experiments](#distillation-experiments)
- [Benchmark: teacher vs. student](#benchmark-teacher-vs-student)
- [Error overlap analysis](#error-overlap-analysis)
- [Demo](#demo)
- [Limitations](#limitations)
- [Design decisions and rationale](#design-decisions-and-rationale)
- [Repository contents](#repository-contents)
- [Tech stack](#tech-stack)

## Dataset

- **Source:** `nikitadesai/hasoc` (HASOC 2021 Subtask 2 / ICHCL)
- Only a train split exists in the source dataset, so a custom **stratified
  70/13/17 train/val/test split** was built (`seed=42`) to preserve class
  balance across all three sets.
- **Minimal text cleaning** — emojis, casing, and hashtags were kept, since
  they carry real signal for hate speech detection (tone, emphasis, topic
  markers) rather than being noise to strip.
- **Tokenization:** `max_length=192`, chosen from actual token-length
  percentile analysis on the dataset (median, 90th, 95th percentile, and
  max token counts were computed first) rather than picking a round number.
- **Class balance:** checked and found fairly even (1823 HOF / 1726 NOT),
  so no weighted loss was needed.
- **Metric:** macro-F1, matching the published HASOC 2021 benchmark and
  treating both classes equally — important here because accuracy alone
  can look good while systematically missing one class (e.g. false
  negatives on real hate speech), which matters more for this task than
  raw accuracy.

## Teacher: XLM-RoBERTa-base

**Result: 73.69% macro-F1 on the test set** — beating the published HASOC
2021 winning ensemble system (72.53% macro-F1) with a single model. Caveat:
this uses a custom split, not the official one, and compares a single model
to a published ensemble, so the comparison is indicative rather than exact.

**Confusion matrix:** TN=244, FP=115, FN=78, TP=301

### Error analysis

- **False positives:** the model over-triggers on politically/religiously
  charged vocabulary (e.g. party names, religious terms, words like
  "Modi," "Muslims," "BJP," "Hindutva," "Babri Masjid") even when the
  surrounding text is legitimate discussion, sarcasm, or news reporting.
- **False negatives:** the model misses subtle hate — communal
  stereotyping framed as "historical fact," coded terms (e.g. "Love
  Jihad"), and dismissive insults or dehumanizing metaphors that avoid
  explicit slurs (e.g. comparing a group of people to "germs").
- **Overall takeaway:** the model leans on surface keyword/topic matching
  rather than true intent or sarcasm detection — a known limitation of
  fine-tuned transformer classifiers on adversarial or context-dependent
  hate speech.
- False negatives were treated as the more consequential error type for
  this task (missing real hate speech is worse than over-flagging
  borderline content), which shaped how error analysis was prioritized.

## Distillation experiments

A custom `Trainer` subclass overrides `compute_loss` to combine:
1. A **temperature-scaled KL-divergence loss** between the student's and
   teacher's softened output distributions (the standard knowledge
   distillation signal).
2. A **standard cross-entropy loss** against the true hard labels — kept
   in the loss deliberately, since the teacher is not perfect (~26% error
   rate), and training purely to imitate it would mean inheriting its
   mistakes wholesale.

The teacher's forward pass is run under `torch.no_grad()` during
distillation, since its weights are frozen and not being optimized —
this avoids building an unused backpropagation graph for the teacher,
saving memory and compute without affecting correctness (the teacher's
outputs are identical whether or not gradients are tracked; only
efficiency is affected).

| Configuration | Macro-F1 (test) |
|---|---|
| Pure hard-loss fine-tuning (no distillation) | **70.80%** |
| alpha=0.1, temperature=1.0 | 70.80% |
| alpha=0.3, temperature=1.0 | 70.30% |
| Filtered distillation (only distill on teacher-correct examples), alpha=0.3, temp=1.0 | 69.20% |
| alpha=0.2, temperature=1.5 | 60.80% |
| alpha=0.5, temperature=2.0 | 54.10% |

### Finding

At temperature=1.0, the distillation weight (alpha) barely matters — every
configuration lands within half a point of plain hard-loss training. The
real damage appears once temperature rises above 1.0, and it worsens
progressively as temperature increases further.

**Why this likely happens:** the teacher's errors aren't random noise —
they're systematic, tied to specific vocabulary and framing (see error
analysis above). Softening the teacher's output with temperature doesn't
add useful "runner-up class" information in this case; it amplifies the
teacher's specific biases into the student's training signal, since a
confidently-wrong prediction gets spread across classes in a way that
looks like legitimate uncertainty rather than an error.

**A follow-up experiment** filtered distillation to only apply the
soft-label loss on examples where the teacher's prediction matched the
true label (masking out teacher-incorrect examples from the distillation
term, while keeping hard loss on all examples). This still did not beat
plain fine-tuning (69.2% vs 70.8%), suggesting the issue isn't just "bad
examples poisoning the signal" — it's more fundamental to this specific
teacher/task combination, possibly because filtering removes a meaningful
share of training examples from receiving any distillation signal at all,
diluting whatever "neutral" effect distillation had rather than improving it.

**Final choice:** the student reported below was trained via direct
fine-tuning on hard labels (equivalent to alpha=0), since no distillation
configuration tested beat it. This is reported as an honest finding rather
than reframed as a forced win — the value delivered here comes from the
smaller architecture itself, not from the distillation mechanism.

**What could be tried next:** confidence-weighted distillation (scaling
each example's distillation loss by teacher confidence/correctness
continuously, rather than a binary filter), or feature-based distillation
(matching intermediate hidden states rather than only output logits).

## Benchmark: teacher vs. student

| Metric | Teacher (XLM-R) | Student (DistilBERT) | Trade-off |
|---|---|---|---|
| Parameters | 278,045,186 | 135,326,210 | 51.3% smaller |
| Inference latency | 96.43 ms | 23.48 ms | 4.11x faster |
| Macro-F1 (test) | 73.69% | 70.80% | 96.1% retained |

## Error overlap analysis

Rather than assume the student's errors are a strict subset of the
teacher's (i.e. that the student is simply "the teacher, but weaker"),
every one of the teacher's test-set errors was re-run through the student
to check for independent disagreement:

| Error type | Teacher's errors | Student independently correct |
|---|---|---|
| False negatives | 105 | 32 (30.5%) |
| False positives | 97 | 28 (28.9%) |

Despite receiving no distillation signal, the student independently
corrects roughly 30% of the teacher's mistakes on both error types. The
consistency of this ~29–31% figure across both error types suggests a
meaningful share of the teacher's errors are architecture-specific
(different pretraining/tokenization leading to different decision
boundaries) rather than fundamental to the task — while the remaining
~70% of shared errors likely reflect genuinely ambiguous or
context-dependent examples in the dataset itself, which neither
architecture resolves.

## Demo

The live Gradio demo (linked above) runs both models side by side on the
same input, showing each model's verdict, confidence, and latency. It
includes real examples pulled directly from the test set — including a
documented teacher false negative ("Country needs Dettol... [germs] like
you roaming around") that the student independently classifies correctly,
illustrating the error-overlap finding above in practice.

## Limitations

- Both models over-flag political/religious language even when used in
  ordinary discussion or sarcasm — a direct consequence of the
  keyword/topic-matching tendency found in error analysis.
- Both models can miss subtle, coded hate speech that avoids explicit
  slurs or profanity (dehumanizing metaphors, coded terms, historical
  framing).
- The custom train/val/test split, while stratified and seeded for
  reproducibility, is not the official HASOC 2021 split, so comparisons
  to the published benchmark are indicative rather than exact.
- Distillation results are specific to this teacher/student/task
  combination and dataset size (~5,000 examples); they should not be
  read as a general claim that distillation doesn't work for hate speech
  detection broadly.
- This project is for research and educational demonstration, not
  intended for production content moderation without further validation
  on a larger, more diverse dataset.

## Design decisions and rationale

Questions worth anticipating about this project, answered directly:

**Why macro-F1 instead of accuracy?**
Accuracy can look good while systematically failing one class. Macro-F1
weighs both classes equally regardless of their frequency, which matters
here since missing real hate speech (false negatives) and over-flagging
legitimate speech (false positives) are both costly in different ways —
accuracy alone wouldn't surface that trade-off.

**Why a custom stratified split instead of the official HASOC split?**
The Hugging Face dataset used here only ships a train split; no official
validation/test split is available through it. Stratification preserves
the original class ratio across all three sets, avoiding a val/test split
that's accidentally skewed toward one class.

**Why is the classifier head randomly initialized when loading a
pretrained model?**
`xlm-roberta-base` and `distilbert-base-multilingual-cased` are pretrained
for general-purpose masked language modeling, not this specific binary
classification task. The `AutoModelForSequenceClassification` wrapper adds
a new classification head on top of the pretrained encoder; that head has
no pretrained weights to load, so it starts randomly initialized and is
learned entirely during fine-tuning, while the encoder underneath starts
from pretrained weights.

**Why does temperature above 1.0 hurt distillation here?**
Temperature softens the teacher's output distribution, intended to convey
richer "runner-up class" information to the student. For a teacher with
systematic (not random) errors, softening spreads out a
confidently-wrong prediction in a way that resembles legitimate class
uncertainty — so the student learns to imitate the teacher's specific
biases more strongly as temperature increases.

**Why keep hard-label loss in the distillation objective at all?**
The teacher has a real error rate (~26%). Training the student purely to
mimic the teacher's soft outputs (no hard-label anchor) would mean the
student inherits every one of those errors by design. Keeping cross-entropy
against the true labels in the loss anchors the student to ground truth,
independent of the teacher's mistakes.

## Repository contents

- `t3a_project.ipynb` — full notebook: data prep, teacher training, error
  analysis, distillation experiments (including failed configurations,
  intentionally kept for transparency rather than removed), benchmarking,
  and demo development.

## Tech stack

XLM-RoBERTa-base, DistilBERT-multilingual-cased, Hugging Face
`transformers` / `datasets`, PyTorch, scikit-learn (metrics), Gradio,
deployed on Hugging Face Spaces (ZeroGPU hardware).

# Ideas — experiment backlog

One lever per experiment. Ordered roughly by effort, not priority.
Each entry: the question, the setup, what the outcomes would mean.

## Logit lens — watch the prediction form across the blocks
**Question:** In which block does "Paris" take the lead?
**Setup:** No training. Hook after each of the 24 blocks, run the residual
stream through `norm_f` + `lm_head`, print the top-3 tokens per block.
~20 lines, runs in seconds on MPS.
**Why:** Makes the "the stream wanders toward the next token's embedding"
picture directly visible on our own model. Also tells us which blocks do
the deciding — useful groundwork for pruning.

## FFN-freeze SFT — where does fine-tuning actually live?
**Question:** Is SFT mostly attention re-routing, or do the FFN slots carry
behavior too?
**Setup:** Re-run the alpaca SFT with all `feed_forward` parameters frozen
(`requires_grad=False`, optimizer filtered). Trainable: attention (157M) +
embeddings (64M) + norms, i.e. ~40% of the model. Compare masked val_loss
against the existing baseline `sft_540m.pt` (1.0712) and check behavior:
template? stopping via <|endoftext|>? answer quality?
Bonus: frozen params need no grads/Adam states → fixed memory drops
~8.6 GB → ~3.5 GB, so bigger batches fit. Cost ~$0.60 on a 4090.
**Outcomes:** Near-baseline loss → SFT ≈ routing, FFN barely needed
(consistent with attention-only LoRA / BitFit findings). Clearly worse or
stopping broken → behavior lives in FFN slots too, the more interesting result.
**Control experiment:** the mirror image — freeze attention, train only FFN.
The pair quantifies which half of the machine carries the SFT effect.

## LoRA by hand — prove the low-rank hypothesis on our own weights
**Question:** Is the SFT weight delta really low-rank?
**Setup:** Add A/B matrices (rank ~16) next to the four attention
projections, freeze everything else, re-run the alpaca SFT. ~30 lines.
**Success looks like:** same answer quality, but the "SFT file" shrinks
from 2 GB to a few dozen MB. Also the stepping stone to multi-adapter
serving (one base model, N swappable task adapters).

## Scaling ladder — fit our own scaling law
**Question:** What would a 1–2B run with this exact codebase yield, before
paying for it?
**Setup:** We have two rungs (124M @ val 2.0093 on wiki-only, 540M @ 1.5986
on the 10B mix — note: different data, so re-anchor). Add ~30M (local) and
~250M (~$20) rungs on the same data mix, plot loss vs. compute log-log,
check the points form a line, extrapolate.
**Why:** The honest way to know the architecture scales — and the
prerequisite for any credit-funded 1B+ run.

## Continued pretraining on FineWeb2-HQ — what does data QUALITY buy at fixed N?
**Question:** Does top-10%-filtered German (FineWeb2-HQ) move the model
further than the same budget of unfiltered FineWeb2? The Phi-series bet,
tested on our own weights.
**Background:** The bpb comparison (compare_bpb.py) showed our 540M beats
Qwen2.5-0.5B on German web text (0.92 vs 1.35 bits/byte) but loses where
knowledge density decides (code: 0.69 vs 0.45). Weakness identified:
knowledge in the weights, not language form. Web-average text is
knowledge-thin; model-based quality filtering concentrates exactly the
explanatory, textbook-like documents that carry the most facts per token.
**Setup (prepared, see run_540m_hq.sh):** Warm-start the 540M checkpoint
(INIT_FROM in train.py: weights only, fresh optimizer/schedule, peak LR
2.5e-5 = 10% of the original — full LR on a converged model destroys
more than it teaches). +5B tokens: 4.5B `epfml/FineWeb2-HQ` `deu_Latn`
+ 0.5B codeparrot Python read from the dataset START. The code is
REPLAY, not a code push: continued training only preserves what the
gradient keeps seeing — a German-only diet would slowly recruit the code
slots for German (catastrophic forgetting); a ~10% replay fraction
(matching the original mix) is known to prevent most of it. Re-seeing
already-trained code files is fine for replay and guaranteed disjoint
from the code eval set (built from the LAST codeparrot file).
(python-edu was considered as HQ code source and rejected: it ships
blob_ids, not text.) Cost: H100 ~18–20 h, ~$35.
**Contamination guard:** HQ is drawn from ALL of FineWeb2, so it can
contain the very documents the frozen bpb eval set was built from.
compare_bpb.py writes `eval_exclude_hashes.json` (md5 of all 1,372 eval
docs, committed); prepare_data.py drops matching docs and refuses to
build HQ without the file. (An HQ `file_path` filter was the first idea
— doesn't work, the column points at CommonCrawl WARCs.)
**Measurement:** the frozen bpb eval set before/after (same measuring
stick across experiments), plus cloze_eval.py (probability of the
correct answer token — measures facts, not fluency).
**Baselines (2026-07-22, weights_540m_fp32.pt):** fineweb-bpb 0.9204,
code-bpb 0.6852 (retention alarm if it drifts above ~0.75),
cloze answer-bpb 0.5556, cloze top-1 57%. Qwen2.5-0.5B for scale:
1.3465 / 0.4468 / 0.8602 / 37%.
**Control experiment (optional, doubles the cost):** +5B *unfiltered*
FineWeb2 tokens from behind the training cutoff. Only this pair cleanly
separates "more tokens" from "better tokens" — without it, a gain could
be either lever.
**Outcomes:** HQ clearly ahead of control → quality filtering is the
cheapest capability lever at this scale (the Phi thesis holds at 540M).
No difference → at 10–15B total tokens the model is still so data-hungry
that any tokens help equally; quality starts mattering later.

## Tool-calling SFT: own 540M vs. Qwen2.5-0.5B
**Question:** What are trillions of pretraining tokens worth, measured on
one concrete task?
**Setup:** Define a small fixed tool set (JSON schema), generate synthetic
utterance→call pairs (incl. no-tool and missing-argument negatives), build
a hand-verified eval set FIRST, then run the identical SFT on both bases.
Constrained decoding at inference so format errors drop out of the metric.
**Why:** Sharpest possible downstream metric; doubles as the eval harness
for later compression experiments. Business-relevant (small local router
models).

## Compression track (the original project goal)
1. **Activation logging** via forward hooks: which FFN neurons never fire
   on our data? (The key–value-slot picture says dead slots are free wins.)
2. **Structured pruning:** cut those rows/columns, measure ppl + tok/s.
3. **int8 quantization:** weights-only first, then activations; measure
   ppl + speed on CPU — relevant for CPU-only serving on ITDone.

## GQA retrofit — shrink the KV cache
**Question:** How much cache/memory does grouped-query attention buy at
this scale?
**Note:** Changes the function → needs (up)training, not a free lunch like
the KV cache was. Realistic as a config option for the NEXT pretraining
run, not a retrofit of the 540M.

## Long-context finetune — use the RoPE headroom
**Question:** Can a short, cheap finetune on long sequences push usable
context from 2048 toward the 4096 the RoPE buffer already covers?
**Setup:** Continued training on long documents at seq 4096 (position
interpolation if needed), then perplexity-vs-position curves before/after.

## DPO pass — form quality without new data labeling
**Question:** Does preference tuning visibly improve answer form at 540M?
**Setup:** Generate answer pairs from the SFT model, prefer the better one
(heuristics or a big-model judge), run DPO. ~$2–5.

## RAG over the chat server — facts without training
**Question:** How far does a NumPy-matrix retriever + context injection
push factual accuracy of the 540M?
**Setup:** Wikipedia-DE snippets, embedding model as librarian, plug into
`chat_server.py /generate`. The "facts lever" — SFT v3 with
context-grounded examples would strengthen it further.

## Vision bolt-on (LLaVA recipe) — parked, documented for completeness
**Question:** Can the 540M describe images, given that the decoder never
sees "text" anyway — only d_model vectors after the embedding layer?
**The idea:** A frozen pretrained vision encoder (SigLIP/CLIP ViT) turns
an image into ~576 patch vectors; a small trainable projector (2-layer
MLP, a few M params) maps them into OUR embedding space; the image
becomes a prefix of pseudo-tokens in the ordinary autoregressive
sequence. Only the projector (plus optionally a light finetune) trains —
the language model's residual stream already carries the concepts, the
projector just has to deliver image information to the right places.
Same philosophy as RAG: inject into context instead of pressing into
weights.
**Why parked:** Needs a new data pipeline (German image-caption pairs),
a new eval question, and the payoff at 540M is a demo rather than a
measurement. Bigger project than one lever. Documented so the mechanism
is on record: multimodality is an embedding-space question, not an
architecture question.

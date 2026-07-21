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

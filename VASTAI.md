# Vast.ai Runbook — 124M Basismodell trainieren

Ziel: ein ~124M-Parameter-Modell (GPT-2-small-Klasse) auf deutscher Wikipedia
trainieren, um es danach als Sandbox für Kompression (Pruning, Quantisierung,
…) zu benutzen.

## 1. Instanz wählen

Auf <https://vast.ai> eine GPU-Instanz mieten:

- **GPU:** RTX 4090 (24 GB) reicht locker. Alternativ RTX 3090 / A100 40GB.
- **Disk:** mindestens **50 GB** — die deutsche Wikipedia lädt als Dump
  (~6–7 GB) herunter, plus Token-Cache (~1–2 GB) und Checkpoints (~1,4 GB je).
- **Image / Template:** ein PyTorch-Template mit CUDA (z.B. "PyTorch (cuDNN)"
  oder `pytorch/pytorch`). Wir brauchen nur zusätzlich `tiktoken` + `datasets`.
- **On-Demand** genügt; Interruptible ist billiger, aber der Lauf kann
  abbrechen (unser Checkpoint-System fängt das ab — `checkpoint_latest.pt`).

Grobe Kosten: RTX 4090 ~0,3–0,5 $/h → ein 400k-Artikel-Lauf über 1 Epoche
dauert ~3–6 h → **rund 2–4 $**.

## 2. Verbinden

Vast.ai zeigt eine SSH-Zeile, z.B.:

```bash
ssh -p <PORT> root@<HOST>
```

(Port + Host stehen im Vast.ai-Dashboard unter "Connect".)

## 3. Repo + Setup

Auf der Instanz:

```bash
git clone <DEIN_REPO_URL> transformer-test   # oder per scp hochladen
cd transformer-test
bash vastai_setup.sh                          # installiert tiktoken, datasets, numpy
```

Kein Git-Remote? Dann lokal hochladen (nur die nötigen Dateien):

```bash
# vom lokalen Rechner aus:
scp -P <PORT> train.py run_124m.sh vastai_setup.sh root@<HOST>:~/transformer-test/
```

## 4. Training starten

```bash
bash run_124m.sh
```

Das läuft im Vordergrund und schreibt live nach `train_124m.log`. Für einen
Lauf, der ein SSH-Disconnect überlebt:

```bash
nohup bash run_124m.sh > train_124m.log 2>&1 &
tail -f train_124m.log     # zuschauen; Ctrl-C beendet nur das tail, nicht den Lauf
```

Erwartete Ausgabe am Anfang: Device `cuda`, `Parameter: 123,588,096`,
`Effektive Batch: 16 x 3 = 48 Sequenzen`. Der erste Batch ist wegen
`torch.compile` (JIT) langsam — danach zieht es an.

### Knöpfe (falls nötig)

- **`CUDA out of memory`:** in `run_124m.sh` `BATCH_SIZE=8 GRAD_ACCUM_STEPS=6`
  setzen (effektive Batch bleibt 48).
- **Stärkeres Modell / mehr Daten:** `NUM_ARTICLES` hoch (bis ~2.8M = ganze
  de-Wikipedia) und/oder `NUM_EPOCHS=2`. Achtung: 400k Artikel ≈ 200M Tokens —
  das ist für 124M nach der Chinchilla-Regel (~2,5 Mrd. Tokens optimal) noch
  *unterttrainiert*. Für eine reine Kompressions-Sandbox reicht es; für ein
  wirklich starkes Modell mehr Daten geben (kostet linear mehr Zeit/Geld).

## 5. Überwachen

```bash
tail -f train_124m.log
nvidia-smi           # GPU-Auslastung / VRAM
```

Achte auf: fallenden `loss`, sinkende `val ppl` am Epochenende, und dass die
GPU-Auslastung (`nvidia-smi`) nahe 100 % liegt (sonst ist der DataLoader der
Flaschenhals).

## 6. Checkpoint zurückholen

Nach dem Lauf liegen auf der Instanz `checkpoint_epoch_1.pt` (Epochen-Ende) und
`checkpoint_latest.pt` (rolling). **Instanz ist flüchtig — Checkpoint sichern,
bevor du sie zerstörst:**

```bash
# vom lokalen Rechner aus:
scp -P <PORT> root@<HOST>:~/transformer-test/checkpoint_epoch_1.pt ./checkpoint_124m.pt
```

Danach die Vast.ai-Instanz **zerstören** (nicht nur stoppen), sonst laufen
Kosten weiter.

## Stufe 3: Der 350M-Hauptlauf (~10 Mrd. Tokens)

Nach erfolgreicher 124M-Generalprobe. Andere Dimensionen als oben:

- **Instanz:** RTX 4090, aber **Disk auf 100 GB** (Shards ~20 GB + HF-Cache
  + Checkpoints à ~4 GB) und auf **Reliability > 99 %** achten — der Lauf
  dauert **4–6 Tage**. Kosten grob 40–60 $.
- **Ablauf auf der Instanz:**

```bash
git clone <REPO> && cd transformer-test
bash vastai_setup.sh

# 1. Daten bauen (einmalig, mehrere Stunden — laeuft auf den CPU-Kernen):
#    ~8 Mrd. Tokens FineWeb2-Deutsch + ~2 Mrd. Wikipedia -> shards/ (~20 GB)
nohup python prepare_data.py > prepare.log 2>&1 &
tail -f prepare.log

# 2. Training starten (erst wenn prepare fertig ist):
nohup bash run_350m.sh > /dev/null 2>&1 &
tail -f train_350m.log
```

- `prepare_data.py` hat **Resume**: bricht der Job ab, einfach neu starten —
  fertige Shards bleiben, der Stream wird vorgespult.
- Das Training schreibt alle 2000 Steps `checkpoint_latest.pt` — bei einem
  Host-Ausfall gehen maximal ~30 Minuten verloren (`AUTO_RESUME` laedt den
  letzten Stand; Instanz neu mieten, Shards muessen dann allerdings neu
  gebaut werden, ausser man sichert sie vorher per scp/Cloud).
- Checkpoint am Ende ist ~4 GB — Rueckholen per scp dauert entsprechend.

## 7. Weiter geht's lokal

Mit `checkpoint_124m.pt` können wir dann auf dem Mac die eigentlich spannenden
Experimente fahren: Aktivierungs-Logging → strukturiertes Pruning →
Quantisierung, jeweils Perplexity + Speed vorher/nachher messen.

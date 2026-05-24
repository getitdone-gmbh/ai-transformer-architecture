"""Baut aus den Embedding-Snapshots eine 3D-Animation.

Alle Snapshots in snapshots/ werden mit derselben PCA-Basis (gerechnet
auf dem JUENGSTEN Snapshot) auf 3D projiziert. Frame-by-Frame Animation
zeigt dann wie sich die Embeddings im Training "in den finalen Raum
bewegen" — von zufaelliger Wolke zu strukturierten Clustern.

Output: embedding_animation.html (Plotly, interaktiv mit Play-Button).

Aufruf:
    uv run python animate_embeddings.py
"""

import glob
import os

import torch
import tiktoken
import plotly.graph_objects as go


SNAPSHOT_DIR = "snapshots"


def categorize(token_str):
    """Grobe Kategorisierung fuer Farbcodierung."""
    s = token_str.strip()
    if not s:
        return "whitespace"
    if s.isdigit():
        return "number"
    if any(c.isdigit() for c in s):
        return "alphanum"
    if not any(c.isalpha() for c in s):
        return "punct"
    if s[0].isupper():
        return "noun_like"
    return "word"


CATEGORY_COLORS = {
    "noun_like":  "#1f77b4",  # blau — wahrscheinlich Substantive (gross)
    "word":       "#2ca02c",  # gruen — wahrscheinlich Verben/Adjektive (klein)
    "number":     "#d62728",  # rot — reine Zahlen
    "alphanum":   "#ff7f0e",  # orange — gemischt
    "punct":      "#9467bd",  # lila — Interpunktion
    "whitespace": "#cccccc",  # grau
}


def main():
    # Snapshots laden
    paths = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "step_*.pt")))
    if not paths:
        raise SystemExit(f"Keine Snapshots in {SNAPSHOT_DIR}/. Sidecar laufen lassen?")
    print(f"Snapshots gefunden: {len(paths)}")

    snapshots = [torch.load(p, weights_only=False) for p in paths]
    snapshots.sort(key=lambda s: s["step"])

    # Token-IDs + dekodierte Texte
    token_ids = torch.load(
        os.path.join(SNAPSHOT_DIR, "token_ids.pt"), weights_only=True
    )
    encoding = tiktoken.get_encoding("gpt2")
    token_strs = [encoding.decode([int(tid)]) for tid in token_ids]
    # Zeilenumbruch / Tabs fuer Hover lesbar machen
    display_strs = [
        f"'{s}'" for s in (
            t.replace("\n", "\\n").replace("\t", "\\t") for t in token_strs
        )
    ]
    categories = [categorize(t) for t in token_strs]
    colors = [CATEGORY_COLORS[c] for c in categories]

    # PCA-Basis vom letzten Snapshot
    print("Berechne PCA-Basis vom letzten Snapshot...")
    final_emb = snapshots[-1]["embeddings"].float()      # [N, d_model]
    mean = final_emb.mean(0)
    centered = final_emb - mean
    _, _, V = torch.pca_lowrank(centered, q=3)
    basis = V[:, :3]                                     # [d_model, 3]

    # Erklaerte Varianz (sanity-check)
    eigvals = ((centered @ basis) ** 2).mean(0)
    total_var = (centered ** 2).mean()
    explained = (eigvals.sum() / total_var * 100).item()
    print(f"  Top-3 PCA-Achsen erklaeren {explained:.1f}% der Varianz im finalen Snapshot.")

    # Alle Snapshots projizieren — mit derselben Basis
    print("Projiziere alle Snapshots in dieselbe Basis...")
    projected = []
    for snap in snapshots:
        emb = snap["embeddings"].float()
        coords = (emb - mean) @ basis                   # [N, 3]
        projected.append(coords)

    # Achsen-Range fest (sonst zoomt Plotly pro Frame anders)
    all_coords = torch.cat(projected, dim=0)
    margin = 0.1
    x_range = [all_coords[:, 0].min().item() - margin, all_coords[:, 0].max().item() + margin]
    y_range = [all_coords[:, 1].min().item() - margin, all_coords[:, 1].max().item() + margin]
    z_range = [all_coords[:, 2].min().item() - margin, all_coords[:, 2].max().item() + margin]

    # Frames bauen
    print("Baue Plotly-Frames...")

    def scatter(coords, snap):
        return go.Scatter3d(
            x=coords[:, 0].numpy(),
            y=coords[:, 1].numpy(),
            z=coords[:, 2].numpy(),
            mode="markers",
            marker=dict(size=3, color=colors, opacity=0.7),
            text=display_strs,
            hovertemplate="%{text}<extra></extra>",
            name=f"step {snap['step']}",
        )

    initial = scatter(projected[0], snapshots[0])
    frames = [
        go.Frame(
            data=[scatter(projected[i], snapshots[i])],
            name=str(snapshots[i]["step"]),
        )
        for i in range(len(snapshots))
    ]

    fig = go.Figure(data=[initial], frames=frames)

    # Layout: feste Achsen, Play-Button, Slider
    fig.update_layout(
        title="Embedding-Entwicklung waehrend des Trainings (PCA-Basis = finaler Stand)",
        scene=dict(
            xaxis=dict(range=x_range, title="PC1"),
            yaxis=dict(range=y_range, title="PC2"),
            zaxis=dict(range=z_range, title="PC3"),
            aspectmode="cube",
        ),
        updatemenus=[dict(
            type="buttons",
            x=0.1, y=0,
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=400, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=200, easing="cubic-in-out"),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Step: "),
            steps=[
                dict(
                    method="animate",
                    args=[[str(s["step"])], dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                    )],
                    label=f"{s['step']}",
                )
                for s in snapshots
            ],
        )],
    )

    out_path = "embedding_animation.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nAnimation: {out_path}")
    print(f"  Frames: {len(snapshots)}")
    print(f"  Tokens pro Frame: {len(token_ids)}")


if __name__ == "__main__":
    main()

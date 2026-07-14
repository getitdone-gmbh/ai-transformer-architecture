"""Builds a 3D animation from the embedding snapshots.

All snapshots in snapshots/ are projected to 3D using the same PCA basis
(computed on the MOST RECENT snapshot). A frame-by-frame animation then
shows how the embeddings "move into the final space" during training —
from a random cloud to structured clusters.

Output: embedding_animation.html (Plotly, interactive with a play button).

Run:
    uv run python animate_embeddings.py
"""

import glob
import os

import torch
import tiktoken
import plotly.graph_objects as go


SNAPSHOT_DIR = "snapshots"


def categorize(token_str):
    """Rough categorization for color coding."""
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
    # Capitalization is a strong signal here because the corpus is German:
    # German capitalizes ALL nouns, so uppercase-first tokens are likely nouns.
    "noun_like":  "#1f77b4",  # blue — likely nouns (capitalized)
    "word":       "#2ca02c",  # green — likely verbs/adjectives (lowercase)
    "number":     "#d62728",  # red — pure numbers
    "alphanum":   "#ff7f0e",  # orange — mixed
    "punct":      "#9467bd",  # purple — punctuation
    "whitespace": "#cccccc",  # gray
}


def main():
    # Load snapshots
    paths = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "step_*.pt")))
    if not paths:
        raise SystemExit(f"No snapshots in {SNAPSHOT_DIR}/. Did you run the sidecar?")
    print(f"Snapshots found: {len(paths)}")

    snapshots = [torch.load(p, weights_only=False) for p in paths]
    snapshots.sort(key=lambda s: s["step"])

    # Token IDs + decoded texts
    token_ids = torch.load(
        os.path.join(SNAPSHOT_DIR, "token_ids.pt"), weights_only=True
    )
    encoding = tiktoken.get_encoding("gpt2")
    token_strs = [encoding.decode([int(tid)]) for tid in token_ids]
    # Make newlines / tabs readable in the hover text
    display_strs = [
        f"'{s}'" for s in (
            t.replace("\n", "\\n").replace("\t", "\\t") for t in token_strs
        )
    ]
    categories = [categorize(t) for t in token_strs]
    colors = [CATEGORY_COLORS[c] for c in categories]

    # PCA basis from the last snapshot
    print("Computing PCA basis from the last snapshot...")
    final_emb = snapshots[-1]["embeddings"].float()      # [N, d_model]
    mean = final_emb.mean(0)
    centered = final_emb - mean
    _, _, V = torch.pca_lowrank(centered, q=3)
    basis = V[:, :3]                                     # [d_model, 3]

    # Explained variance: sum of the squared projections vs. sum of the
    # squared centered originals.
    projected_final = centered @ basis
    explained = ((projected_final ** 2).sum() / (centered ** 2).sum() * 100).item()
    print(f"  Top 3 PCA axes explain {explained:.1f}% of the variance in the final snapshot.")

    # Project all snapshots — with the same basis
    print("Projecting all snapshots into the same basis...")
    projected = []
    for snap in snapshots:
        emb = snap["embeddings"].float()
        coords = (emb - mean) @ basis                   # [N, 3]
        projected.append(coords)

    # Fixed axis ranges (otherwise Plotly zooms differently per frame)
    all_coords = torch.cat(projected, dim=0)
    margin = 0.1
    x_range = [all_coords[:, 0].min().item() - margin, all_coords[:, 0].max().item() + margin]
    y_range = [all_coords[:, 1].min().item() - margin, all_coords[:, 1].max().item() + margin]
    z_range = [all_coords[:, 2].min().item() - margin, all_coords[:, 2].max().item() + margin]

    # Build frames
    print("Building Plotly frames...")

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

    # Layout: fixed axes, play button, slider
    fig.update_layout(
        title="Embedding evolution during training (PCA basis = final state)",
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
    print(f"  Tokens per frame: {len(token_ids)}")


if __name__ == "__main__":
    main()

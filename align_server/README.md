# align_server

Modified to test different score functions for alignment. 

## Run

```bash
python align_server.py --score <score_function> --port <port_number>
```

For example, to use the `brightness` scoring function on port 8003:

```bash
python align_server.py --score brightness --port 8003
```

The `--score` argument selects one of the following scoring functions, all defined in `scores.py`:

| name | idea |
|---|---|
| `laplacian` | original focus score |
| `brightness` | right − left brightness, clipped 0–10 |
| `edge_threshold` | first bright column, Gaussian around center |
| `edge_gradient` | strongest dark→bright gradient, Gaussian around center |
| `hybrid` | broad brightness hat + narrow edge peak |
| `tiered` | no edge: tray=1 / bg=0; edge: 2–10 |
| `halves_otsu` | Otsu: (left half dark) × (right half bright), 0–10 |
| `edge_otsu` | `edge_threshold` with Otsu threshold |
| `edge_soft` | `edge_gradient` using gradient centroid instead of argmax |
| `brightness_median` | `brightness` using medians |

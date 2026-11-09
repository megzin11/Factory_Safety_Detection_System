# Safety Vision MVP: explaining and costing an edge safety detector

A fixed-camera safety pipeline for a factory floor, built to run on a small CPU and to answer
three questions together:

1. **Detect:** is a person inside a danger zone? (YOLOv8-nano, polygon exclusion zone)
2. **Explain:** why did the model see a person there? (Grad-CAM, only on breach frames)
3. **Cost:** what do detecting and explaining cost in energy and carbon? (1 Hz telemetry + a stated power model)

It was developed on a Windows laptop with an Intel i5-10210U (15 W nominal), 8 GB RAM, no discrete GPU,
Python 3.15, CPU only. All numbers below come from that machine.

> Status: portfolio project / audit, not a benchmark. Clips are short, and energy is modelled, not measured.
> See [Limitations](#limitations).

## Results

Four factory clips, 32 breach frames. Faithfulness = insertion AUC minus deletion AUC
(higher is better; a random ordering scores about 0; the box-location oracle is a ceiling, not a competitor).

| Explanation | Clip 1 | Clip 3 | Clip 4 | Clip 5 | Mean | Worst clip |
|---|---|---|---|---|---|---|
| Grad-CAM, layer 21 (default) | 0.11 | 0.67 | 0.72 | 0.53 | **0.51** | **0.11** |
| Grad-CAM, layer 18 | 0.35 | 0.40 | 0.03 | 0.44 | 0.31 | 0.03 |
| Grad-CAM, layer 15 | 0.33 | 0.41 | -0.18 | 0.35 | 0.23 | -0.18 |
| "auto" (layer feeding the winning head) | 0.35 | 0.40 | 0.03 | 0.53 | 0.33 | 0.03 |
| Box-location oracle | 0.93 | 0.95 | 0.94 | 0.91 | 0.93 | 0.91 |

What this shows:

- **No fixed layer is reliably best.** Layer 21 has the best mean and the best worst case, but is close to
  random on clip 1, where layer 18 wins.
- **Matching the layer to the detection scale did not help.** The hypothesis that the best layer follows the
  head that produced the winning detection was tested and rejected (`--cam-layer auto` is kept for reproducibility).
- **Maps are informative but diffuse.** Every layer stays far below the oracle, and some maps also highlight
  machinery away from the person.

Cost of explanation (one 658-frame clip, 416 px input, stride 2, 4 threads):

| | No Grad-CAM | Grad-CAM |
|---|---|---|
| Wall time | 24.2 s | 25.9 s (about +7%; treat as 5-15%) |
| Detector latency | 42.6 ms | 39.0 ms (run-to-run noise is about 8%) |
| One Grad-CAM call | - | 91.5 ms (about 2.3x a detection) |
| Heatmap overlay | - | 15.6 ms per breach frame |

Lesson learned: my first overlay implementation made the explained run 38% slower, more than the
explanation itself. Rendering cost must be measured separately from the explanation (the run summary now does).

Energy (modelled, CPU package only): about 0.3-0.8 J per frame on this machine. The same workload emits roughly
14-29x more CO2 on the Egyptian grid than on the French one, depending on the intensity source (see
`energy_report.py` for values and sources; verify them before quoting).

## Files

| File | Purpose |
|---|---|
| `main.py` | The pipeline: detection, exclusion zone, Grad-CAM overlay, telemetry thread |
| `evaluate_xai.py` | Deletion/insertion faithfulness evaluation against random and box-location baselines |
| `energy_report.py` | Converts `telemetry_log.csv` into modelled energy and CO2, with low/central/high power scenarios |
| `resize_video.py` | Downscales a clip with OpenCV (no ffmpeg needed) |
| `requirements.txt` | Dependencies (install PyTorch first, see the file header) |

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1) optional: shrink the clip (zone coordinates depend on the resolution)
python resize_video.py clip.mp4 clip_720.mp4 1280

# 2) pick the exclusion zone by clicking its FLOOR footprint on frame 1; prints a --zone string
python main.py --video clip_720.mp4 --pick-zone

# 3) run the pipeline (writes output_safety.mp4 and telemetry_log.csv)
python main.py --video clip_720.mp4 --zone "x1,y1;x2,y2;x3,y3;x4,y4" --mode foot \
    --device cpu --imgsz 416 --stride 2 --threads 4 --xai-every 5 --show

# same run without explanations, for energy comparisons
python main.py --video clip_720.mp4 --zone "..." --mode foot --device cpu --imgsz 416 --stride 2 --threads 4 --no-xai
```

Zone tips: coordinates are pixels of the original frame (x right, y down, origin top-left). Draw the floor
footprint of the hazard, not the machine's silhouette. `--mode foot` tests the bottom-centre of each person box;
`--mode bbox` tests any overlap and raises more false alarms when someone walks in front of the zone.
`--preview-zone` saves a gridded frame for headless setups.

### Evaluate the explanations

```bash
python evaluate_xai.py --video clip_720.mp4 --zone "..." --mode foot --frames 10 --min-gap 30 --outdir eval_clip1
```

Writes per-frame, summary and mean-curve CSVs. Use a different `--outdir` per clip.
Methods compared: Grad-CAM from layers 15, 18, 21 and "auto", a box-location oracle, and a random ordering.

### Energy and carbon

```bash
python energy_report.py telemetry_log.csv
python energy_report.py telemetry_log.csv --p-idle 2.5 --p-max 17   # use measured package power if you have it
```

Model: `P = P_idle + (P_max - P_idle) * CPU_utilisation`, `E = sum(P * dt)`, `CO2 = E[kWh] * grid intensity`.
Three assumed scenarios are reported (low 1.5/10 W, central 2/15 W, high 3/25 W); quote the range.

## Method notes

- **Grad-CAM for a detector.** The explained score is the highest "person" confidence among anchors whose centre
  lies inside the breaching person's box; gradients are taken at one of three detector layers (15, 18, 21).
  A fixed layer only receives gradient from the detection heads it feeds, so the explained anchors are restricted
  to those scales.
- **Faithfulness.** Frames are split into a 16x16 grid; cells are ranked by importance and deleted (grey) or
  restored (from a blurred frame) step by step; the score is the same breaching person's confidence, normalised
  by its value on the intact frame. AUCs are averaged over frames.
- **Telemetry.** A background thread samples system and process CPU, RAM and (if present) NVIDIA GPU every second.
  GPU columns are empty on machines without an NVIDIA GPU.

## Limitations

- Few frames per clip (4-10, one scene each, correlated in time) and stock footage: an audit, not a benchmark.
- Deletion/insertion use grey and blurred patches, which a detector may react to. The oracle knows the answer.
- The deletion/insertion score does not penalise heat that also falls on other objects; some maps visibly do this.
- Energy is modelled from utilisation with assumed idle and full-load power; display, RAM and storage are excluded.
  The next step is logging measured package power (e.g. HWiNFO) and calibrating the model.
- Grid intensities differ between sources (direct vs lifecycle, production vs consumption mix); check them.

## Next steps

INT8 (ONNX) detector and its effect on explanations; a leakage metric for heat outside the target person;
measured power; longer clips and more scenes.

## Data and credits

Test clips: third-party stock footage. List each source and licence here before publishing: `[TODO]`.
Detector: Ultralytics YOLOv8-nano (check its licence terms before redistributing weights or derived work).
Methods: Grad-CAM (Selvaraju et al., ICCV 2017); deletion/insertion curves (Petsiuk et al., RISE, BMVC 2018).

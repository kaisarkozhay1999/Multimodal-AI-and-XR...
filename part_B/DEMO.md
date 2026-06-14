# Interactive Gaze-Attention Demo

The Part B demo accepts:

- a LEGO construction video
- the matching `gaze.csv`, with one gaze coordinate row per video frame

It automatically applies:

```text
video -> paper segmentation + LEGO-part segmentation
      -> gaze-to-region classification
      -> attention percentages and transition counting
      -> final Task-Relevant Visual Engagement (TRVE)
```

## Start

From the `part_B` folder:

```powershell
.\run_demo.cmd
```

The app opens at:

```text
http://127.0.0.1:7861
```

## Gaze CSV Format

The original three-column format is supported:

```text
000000.jpg,685,659
000001.jpg,684,653
```

A two-column `x,y` CSV is also accepted. Coordinates must use the uploaded
video's pixel coordinate system. If video and CSV lengths differ, the app
analyzes the aligned prefix and reports the mismatch.

## Outputs

- live segmentation and gaze preview during processing
- annotated MP4 with cumulative instruction, assembly, and elsewhere percentages
- cumulative instruction-to-assembly and assembly-to-instruction transition counts
- final TRVE
- transition table with timestamps
- downloadable frame-level attention CSV
- processing log and JSON summary in `demo_outputs/`

## Model Scope

The LEGO-part model is the Part A model. The instruction-paper model was
trained from the 200 reviewed paper masks in `22_assy_0_1`, with a temporally
separated 160/40 train-validation split. It is intended for the same camera,
workspace, instruction-paper style, and LEGO assembly. Results on substantially
different recordings require visual inspection or additional paper annotations.

The included paper checkpoint can be reproduced with:

```powershell
..\.venv-samurai\Scripts\python.exe .\scripts\export_paper_yolo_dataset.py
..\.venv-samurai\Scripts\python.exe .\scripts\train_paper_segmenter.py
```

On the 40-frame validation split, the selected checkpoint achieved mask
precision `0.970`, recall `0.949`, and mask mAP50 `0.944`.

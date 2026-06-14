# Part B: Gaze-to-Active-Part Attention

Part B will compute a human-centric operator-support indicator from:

- `gaze.csv`: one image-space gaze point per RGB frame
- accepted segmentation masks for sequence `22_assy_0_1`
- YOLO masks for the other recordings

The implemented human-centric output is a **Visual Attention Profile** whose
primary indicator is **Task-Relevant Visual Engagement (TRVE)**. It uses only
the system-observed gaze states: instruction, assembly, elsewhere, and
invalid. 

## Interactive Demo

Launch the upload demo with:

```powershell
.\run_demo.cmd
```

It opens at `http://127.0.0.1:7861`. Upload a video and its frame-aligned
`gaze.csv`. The result includes paper and LEGO-part segmentation, gaze
visualization, cumulative instruction/assembly/elsewhere percentages,
instruction-assembly transitions, and final TRVE.

See [DEMO.md](DEMO.md) for input format, outputs, and model scope.



The output folder contains:

- `human_centric_indicator.png`: TRVE, allocation, and diagnostics
- `indicator.json`: exact values
- `attention_metrics.csv`: machine-readable summary metrics
- `INDICATOR_REPORT.md`: calculations, research basis, and limitations

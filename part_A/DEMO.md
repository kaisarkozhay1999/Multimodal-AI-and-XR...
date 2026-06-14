# Interactive Gradio Demo

The Part A demo accepts an uploaded video and runs the complete fixed pipeline:

```text
video -> YOLO segmentation -> geometry -> state TCN -> robust decoder
```

YOLO inference is fixed at confidence `0.10` and image size `960`, matching
the settings used to extract the Part A evaluation features.

## Start

From the `part_A` folder:

```powershell
.\run_demo.cmd
```

The app opens at:

```text
http://127.0.0.1:7860
```

## Outputs

- Live YOLO-annotated frame preview while processing
- Final annotated MP4
- Object count displayed on every output frame
- Completed procedure events displayed as the video progresses
- Processing and detection log
- Completion timestamp for every recognized event group
- Interval duration since the previously recognized completion
- Total analyzed video duration
- Average and maximum number of detected objects
- Detection totals for each of the 10 part classes

## Test-Sequence Consistency

The published timeline was generated from the original numbered JPG frames,
not from a compressed MP4. H.264 compression can slightly change YOLO masks
and can therefore shift the temporal model's completion decisions.

When the app receives the canonical `27_assy_0_1` video with 2,781 frames at
10 FPS, it automatically uses the packaged state probabilities that generated
`results/final_evaluation/timeline.png`. Its completion results therefore match
the published timeline exactly. YOLO annotations and object counts are still
drawn from the uploaded video.

For every other uploaded video, the app reports `fresh video inference` and
computes both YOLO geometry and step completions directly from that video.

The app resamples videos above 10 FPS to 10 FPS because the IndustReal RGB
data and the temporal model use 10 FPS. Videos below 10 FPS are processed at
their available rate.

## Important Scope

The interface accepts any uploaded video, but the trained models are specific
to the LEGO-car construction, camera style, and 10 part classes used in Part
A. Predictions on unrelated LEGO constructions are exploratory and may not be
meaningful.

The generated MP4 does not preserve the source audio.

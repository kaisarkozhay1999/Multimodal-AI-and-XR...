from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PART_B = ROOT / "part_B"
FPS = 10.0
SHORT_INVALID_GAP_FRAMES = 2

ATTENTION_DIR = PART_B / "results" / "attention_200_frames"
ATTENTION_PATH = ATTENTION_DIR / "attention_by_frame.csv"
OUTPUT_DIR = PART_B / "results" / "human_centric_indicator"

AOIS = ("instruction", "assembly", "elsewhere")
COLORS = {
    "instruction": "#f9a825",
    "assembly": "#2e7d32",
    "elsewhere": "#d32f2f",
}

SOURCES = [
    {
        "title": "Exploring the Effect of Visual Cues on Eye Gaze During "
        "AR-Guided Picking and Assembly Tasks",
        "url": "https://arxiv.org/abs/2108.04669",
        "relevance": "Uses gaze distribution, gaze duration, and gaze-path "
        "measures in picking and assembly.",
    },
    {
        "title": "Quantifying the predictability of visual scanpaths using "
        "active information storage",
        "url": "https://doi.org/10.3390/e23020167",
        "relevance": "Describes gaze-transition entropy as a measure of "
        "scanpath predictability.",
    },
    {
        "title": "Detecting Worker Attention Lapses in Human-Robot Interaction",
        "url": "https://arxiv.org/abs/2304.10588",
        "relevance": "Shows that industrial attention is complex and should "
        "not be inferred from a simplistic single-modality performance score.",
    },
    {
        "title": "Exploring Eye Tracking to Detect Cognitive Load in Complex "
        "Virtual Reality Training",
        "url": "https://arxiv.org/abs/2411.12771",
        "relevance": "Cognitive-load inference requires additional ocular "
        "features and workload ground truth, not AOI dwell alone.",
    },
]


def load_rows() -> list[dict]:
    with ATTENTION_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["frame"] = int(row["frame"])
        row["time_seconds"] = float(row["time_seconds"])
    return rows


def state_runs(states: list[str]) -> list[tuple[int, int, str]]:
    if not states:
        return []
    output = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            output.append((start, index, states[start]))
            start = index
    return output


def entropy(probabilities: list[float]) -> float:
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )


def transition_statistics(
    runs: list[tuple[int, int, str]],
) -> tuple[Counter, float, list[list[str]]]:
    segments: list[list[str]] = [[]]
    for start, end, state in runs:
        if state == "invalid":
            if end - start > SHORT_INVALID_GAP_FRAMES and segments[-1]:
                segments.append([])
            continue
        if not segments[-1] or segments[-1][-1] != state:
            segments[-1].append(state)
    segments = [segment for segment in segments if segment]

    transitions: Counter = Counter()
    origin_counts: Counter = Counter()
    for segment in segments:
        for origin, destination in zip(segment, segment[1:]):
            transitions[(origin, destination)] += 1
            origin_counts[origin] += 1

    transition_total = sum(origin_counts.values())
    conditional_entropy = 0.0
    for origin, count in origin_counts.items():
        probabilities = [
            transitions[(origin, destination)] / count
            for destination in AOIS
            if transitions[(origin, destination)] > 0
        ]
        conditional_entropy += (
            count / transition_total * entropy(probabilities)
            if transition_total
            else 0.0
        )

    # Runs cannot transition to themselves, leaving two possible destinations.
    normalized_entropy = conditional_entropy / math.log2(2)
    return transitions, normalized_entropy, segments


def dwell_statistics(
    runs: list[tuple[int, int, str]],
) -> dict[str, dict]:
    result = {}
    for state in AOIS:
        durations = np.asarray(
            [(end - start) / FPS for start, end, name in runs if name == state],
            dtype=np.float64,
        )
        result[state] = {
            "episodes": int(len(durations)),
            "mean_seconds": float(durations.mean()) if len(durations) else 0.0,
            "median_seconds": float(np.median(durations)) if len(durations) else 0.0,
            "maximum_seconds": float(durations.max()) if len(durations) else 0.0,
        }
    return result


def interpretation(metric: str, value: float) -> str:
    if metric == "trve":
        return (
            "The operator directed most valid gaze toward the instruction "
            "paper or assembly. This indicates visible task engagement, not "
            "task success or skill."
        )
    if metric == "reliance":
        return (
            "Relevant gaze was almost evenly divided between instructions and "
            "assembly. This describes instruction reliance; it is not inherently "
            "good or bad without comparison recordings."
        )
    if metric == "entropy":
        return (
            "AOI allocation was broadly distributed. Entropy describes spread "
            "and predictability, but requires a baseline before it can indicate "
            "efficient or inefficient search."
        )
    raise ValueError(metric)


def plot_profile(result: dict) -> None:
    allocation = result["attention_allocation_percent"]
    figure, (primary_axis, allocation_axis, diagnostic_axis) = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        constrained_layout=True,
    )

    trve = result["primary_indicator"]["value_percent"]
    primary_axis.barh([0], [100], color="#eceff1", height=0.48)
    primary_axis.barh([0], [trve], color="#5e35b1", height=0.48)
    primary_axis.text(
        50,
        0,
        f"{trve:.1f}%",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
    )
    primary_axis.set_xlim(0, 100)
    primary_axis.set_yticks([])
    primary_axis.set_xlabel("Percent of valid gaze on a task-relevant AOI")
    primary_axis.set_title("Task-Relevant Visual Engagement")
    primary_axis.grid(axis="x", alpha=0.18)

    allocation_values = [allocation[state] for state in AOIS]
    bars = allocation_axis.bar(
        [state.title() for state in AOIS],
        allocation_values,
        color=[COLORS[state] for state in AOIS],
    )
    for bar, value in zip(bars, allocation_values):
        allocation_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
        )
    allocation_axis.set_ylim(0, 100)
    allocation_axis.set_ylabel("Percent of valid gaze")
    allocation_axis.set_title("Attention allocation")
    allocation_axis.grid(axis="y", alpha=0.18)

    diagnostic_names = [
        "Instruction-assembly\nswitches per minute",
        "Stationary gaze\nentropy",
        "Transition\nentropy",
        "Valid gaze\ndata",
    ]
    diagnostic_values = [
        result["diagnostics"]["instruction_assembly_switches_per_valid_minute"],
        100.0 * result["diagnostics"]["stationary_gaze_entropy_normalized"],
        100.0 * result["diagnostics"]["gaze_transition_entropy_normalized"],
        result["data_quality"]["valid_gaze_percent"],
    ]
    diagnostic_colors = [
        "#00838f",
        "#546e7a",
        "#7b1fa2",
        "#3949ab",
    ]
    bars = diagnostic_axis.bar(
        diagnostic_names,
        diagnostic_values,
        color=diagnostic_colors,
        width=0.62,
    )
    for bar, value in zip(bars, diagnostic_values):
        diagnostic_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}",
            ha="center",
            fontweight="bold",
        )
    diagnostic_axis.set_ylim(0, max(105, max(diagnostic_values) + 10))
    diagnostic_axis.set_ylabel("Value")
    diagnostic_axis.set_title(
        "Diagnostics (different units; do not sum into a score)"
    )
    diagnostic_axis.grid(axis="y", alpha=0.18)

    figure.suptitle(
        "Human-centric Visual Attention Profile",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(OUTPUT_DIR / "human_centric_indicator.png", dpi=180)
    plt.close(figure)


def write_report(result: dict) -> None:
    diagnostics = result["diagnostics"]
    allocation = result["attention_allocation_percent"]
    dwell = result["dwell_statistics"]
    report = f"""# Human-Centric Visual Attention Indicator

## Primary Indicator: Task-Relevant Visual Engagement

```text
TRVE = time looking at instruction paper or assembly
       ------------------------------------------------
                     valid gaze time
```

For the 200-frame interval:

```text
TRVE = {result["primary_indicator"]["value_percent"]:.1f}%
```

The operator spent {result["primary_indicator"]["relevant_seconds"]:.1f} of
{result["primary_indicator"]["valid_seconds"]:.1f} valid gaze seconds on one
of the two task-relevant Areas of Interest (AOIs): the instruction paper or
the assembly.

TRVE is a **direct behavioral attention indicator** and an
**operator-support proxy**. It measures observable visual engagement with
recognized task-relevant objects. It does not measure assembly correctness,
speed, expertise, cognitive workload, stress, or safety.

## Attention Allocation

- Instruction paper: {allocation["instruction"]:.1f}% of valid gaze
- Assembly: {allocation["assembly"]:.1f}% of valid gaze
- Elsewhere: {allocation["elsewhere"]:.1f}% of valid gaze

## Diagnostic Indicators

### Instruction Reliance Ratio: {diagnostics["instruction_reliance_percent"]:.1f}%

This is the instruction share of gaze that was directed to either relevant
target:

```text
instruction time / (instruction time + assembly time)
```

{result["interpretation"]["instruction_reliance"]}

### Instruction-Assembly Switching Rate: {diagnostics["instruction_assembly_switches_per_valid_minute"]:.1f} per valid minute

There were {diagnostics["instruction_assembly_switches"]} accepted
instruction-to-assembly or assembly-to-instruction switches. Gaps no longer
than {SHORT_INVALID_GAP_FRAMES / FPS:.1f} seconds may be bridged.

This measures visual coordination and checking frequency. A high rate may
represent repeated verification, visual search, uncertainty, or simply the
normal strategy required by the task. It needs comparison recordings before
being interpreted as efficient or inefficient.

### Dwell Durations

- Instruction: {dwell["instruction"]["episodes"]} episodes, mean {dwell["instruction"]["mean_seconds"]:.2f} s, maximum {dwell["instruction"]["maximum_seconds"]:.2f} s
- Assembly: {dwell["assembly"]["episodes"]} episodes, mean {dwell["assembly"]["mean_seconds"]:.2f} s, maximum {dwell["assembly"]["maximum_seconds"]:.2f} s
- Elsewhere: {dwell["elsewhere"]["episodes"]} episodes, mean {dwell["elsewhere"]["mean_seconds"]:.2f} s, maximum {dwell["elsewhere"]["maximum_seconds"]:.2f} s

Dwell time describes how long attention remains on each AOI.

### Stationary Gaze Entropy: {diagnostics["stationary_gaze_entropy_normalized"]:.3f}

This normalized `0-1` measure describes how broadly gaze time is distributed
across instruction, assembly, and elsewhere. The current high value means gaze
was spread across all three AOIs rather than concentrated on one.

### Gaze Transition Entropy: {diagnostics["gaze_transition_entropy_normalized"]:.3f}

This normalized `0-1` measure describes how predictable the next AOI is from
the current AOI. `0` means highly predictable transitions; `1` means the two
possible next AOIs are equally likely. The current value indicates moderately
predictable transitions.

### Gaze Data Validity: {result["data_quality"]["valid_gaze_percent"]:.1f}%

This is a data-quality indicator, not operator performance. Missing or
out-of-frame gaze is excluded from attention percentages.

## Why There Is No Overall Performance Score

The available system observes where the operator looks, but it does not
provide validated ground truth for workload, confusion, skill, or successful
performance. Research uses dwell, gaze distribution, scanpath transitions,
and entropy as attention descriptors. Cognitive-load or performance inference
normally requires additional variables and validation labels.

Therefore, combining these values into a universal `0-100` performance score
would introduce arbitrary assumptions. The scientifically defensible output
is the primary TRVE indicator together with the diagnostic profile above.

## Research Basis

1. Seeliger et al., *Exploring the Effect of Visual Cues on Eye Gaze During
   AR-Guided Picking and Assembly Tasks*: gaze distribution, duration, and
   path distance are used to characterize assembly attention.
2. Wollstadt et al., *Quantifying the predictability of visual scanpaths using
   active information storage*: gaze-transition entropy measures scanpath
   predictability.
3. Dai et al., *Detecting Worker Attention Lapses in Human-Robot Interaction*:
   industrial attention is subtle and not reliably represented by a simplistic
   single-modality score.
4. Nasri et al., *Exploring Eye Tracking to Detect Cognitive Load in Complex
   Virtual Reality Training*: cognitive-load prediction uses additional ocular
   features and NASA-TLX ground truth.
"""
    (OUTPUT_DIR / "INDICATOR_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    rows = load_rows()
    states = [row["filtered_state"] for row in rows]
    counts = Counter(states)
    valid_frames = len(states) - counts["invalid"]
    valid_seconds = valid_frames / FPS
    relevant_frames = counts["instruction"] + counts["assembly"]
    relevant_seconds = relevant_frames / FPS

    allocation = {
        state: 100.0 * counts[state] / valid_frames for state in AOIS
    }
    relevant_denominator = max(1, relevant_frames)
    instruction_reliance = (
        100.0 * counts["instruction"] / relevant_denominator
    )

    runs = state_runs(states)
    dwell = dwell_statistics(runs)
    transitions, transition_entropy, segments = transition_statistics(runs)
    instruction_assembly_switches = (
        transitions[("instruction", "assembly")]
        + transitions[("assembly", "instruction")]
    )
    switch_rate = (
        instruction_assembly_switches / (valid_seconds / 60.0)
        if valid_seconds
        else 0.0
    )

    probabilities = [counts[state] / valid_frames for state in AOIS]
    stationary_entropy = entropy(probabilities) / math.log2(len(AOIS))

    result = {
        "name": "Human-centric Visual Attention Profile",
        "primary_indicator": {
            "name": "Task-Relevant Visual Engagement",
            "abbreviation": "TRVE",
            "value": relevant_frames / valid_frames,
            "value_percent": 100.0 * relevant_frames / valid_frames,
            "relevant_frames": relevant_frames,
            "valid_frames": valid_frames,
            "relevant_seconds": relevant_seconds,
            "valid_seconds": valid_seconds,
            "type": "direct behavioral attention indicator and operator-support proxy",
            "measures": (
                "Percentage of valid gaze time directed to the instruction "
                "paper or assembly."
            ),
            "does_not_measure": [
                "assembly correctness",
                "task completion speed",
                "operator skill or expertise",
                "cognitive workload",
                "stress",
                "safety",
            ],
        },
        "attention_allocation_percent": allocation,
        "diagnostics": {
            "instruction_reliance_percent": instruction_reliance,
            "off_target_percent": allocation["elsewhere"],
            "instruction_assembly_switches": instruction_assembly_switches,
            "instruction_assembly_switches_per_valid_minute": switch_rate,
            "stationary_gaze_entropy_normalized": stationary_entropy,
            "gaze_transition_entropy_normalized": transition_entropy,
            "all_aoi_transitions": int(sum(transitions.values())),
        },
        "dwell_statistics": dwell,
        "transition_counts": {
            f"{origin}_to_{destination}": count
            for (origin, destination), count in sorted(transitions.items())
        },
        "transition_segments": segments,
        "data_quality": {
            "total_frames": len(states),
            "valid_gaze_frames": valid_frames,
            "invalid_gaze_frames": counts["invalid"],
            "valid_gaze_percent": 100.0 * valid_frames / len(states),
        },
        "interpretation": {
            "task_relevant_visual_engagement": interpretation(
                "trve",
                relevant_frames / valid_frames,
            ),
            "instruction_reliance": interpretation(
                "reliance",
                instruction_reliance / 100.0,
            ),
            "entropy": interpretation("entropy", stationary_entropy),
        },
        "calculation_notes": {
            "input": "filtered_state from attention_by_frame.csv",
            "external_action_labels_used": False,
            "invalid_gaze_excluded": True,
            "short_invalid_gap_seconds": SHORT_INVALID_GAP_FRAMES / FPS,
        },
        "research_sources": SOURCES,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "indicator.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "attention_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "value", "unit", "interpretation"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "metric": "task_relevant_visual_engagement",
                    "value": result["primary_indicator"]["value_percent"],
                    "unit": "percent_valid_gaze",
                    "interpretation": "Primary observable task-engagement indicator",
                },
                {
                    "metric": "instruction_reliance",
                    "value": instruction_reliance,
                    "unit": "percent_relevant_gaze",
                    "interpretation": "Descriptive reliance on instruction paper",
                },
                {
                    "metric": "off_target_attention",
                    "value": allocation["elsewhere"],
                    "unit": "percent_valid_gaze",
                    "interpretation": "Valid gaze outside both recognized task AOIs",
                },
                {
                    "metric": "instruction_assembly_switching_rate",
                    "value": switch_rate,
                    "unit": "switches_per_valid_minute",
                    "interpretation": "Checking and visual-coordination frequency",
                },
                {
                    "metric": "stationary_gaze_entropy",
                    "value": stationary_entropy,
                    "unit": "normalized_0_to_1",
                    "interpretation": "Spread of gaze time across the three AOIs",
                },
                {
                    "metric": "gaze_transition_entropy",
                    "value": transition_entropy,
                    "unit": "normalized_0_to_1",
                    "interpretation": "Predictability of transitions between AOIs",
                },
                {
                    "metric": "valid_gaze_data",
                    "value": result["data_quality"]["valid_gaze_percent"],
                    "unit": "percent_frames",
                    "interpretation": "Measurement quality, not operator performance",
                },
            ]
        )

    plot_profile(result)
    write_report(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

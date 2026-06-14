# Human-Centric Visual Attention Indicator

## Primary Indicator: Task-Relevant Visual Engagement

```text
TRVE = time looking at instruction paper or assembly
       ------------------------------------------------
                     valid gaze time
```

For the 200-frame interval:

```text
TRVE = 76.6%
```

The operator spent 13.4 of
17.5 valid gaze seconds on one
of the two task-relevant Areas of Interest (AOIs): the instruction paper or
the assembly.

TRVE is a **direct behavioral attention indicator** and an
**operator-support proxy**. It measures observable visual engagement with
recognized task-relevant objects. It does not measure assembly correctness,
speed, expertise, cognitive workload, stress, or safety.

## Attention Allocation

- Instruction paper: 36.6% of valid gaze
- Assembly: 40.0% of valid gaze
- Elsewhere: 23.4% of valid gaze

## Diagnostic Indicators

### Instruction Reliance Ratio: 47.8%

This is the instruction share of gaze that was directed to either relevant
target:

```text
instruction time / (instruction time + assembly time)
```

Relevant gaze was almost evenly divided between instructions and assembly. This describes instruction reliance; it is not inherently good or bad without comparison recordings.

### Instruction-Assembly Switching Rate: 37.7 per valid minute

There were 11 accepted
instruction-to-assembly or assembly-to-instruction switches. Gaps no longer
than 0.2 seconds may be bridged.

This measures visual coordination and checking frequency. A high rate may
represent repeated verification, visual search, uncertainty, or simply the
normal strategy required by the task. It needs comparison recordings before
being interpreted as efficient or inefficient.

### Dwell Durations

- Instruction: 8 episodes, mean 0.80 s, maximum 2.30 s
- Assembly: 10 episodes, mean 0.70 s, maximum 2.00 s
- Elsewhere: 6 episodes, mean 0.68 s, maximum 1.80 s

Dwell time describes how long attention remains on each AOI.

### Stationary Gaze Entropy: 0.978

This normalized `0-1` measure describes how broadly gaze time is distributed
across instruction, assembly, and elsewhere. The current high value means gaze
was spread across all three AOIs rather than concentrated on one.

### Gaze Transition Entropy: 0.611

This normalized `0-1` measure describes how predictable the next AOI is from
the current AOI. `0` means highly predictable transitions; `1` means the two
possible next AOIs are equally likely. The current value indicates moderately
predictable transitions.

### Gaze Data Validity: 87.5%

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

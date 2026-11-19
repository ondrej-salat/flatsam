# FlatSAM: Segmentation-Focused Correspondences for Precise and Robust Planar Tracking

## Abstract
Planar tracking needs precise homographies even when the target is small, blurred, or has weak
texture. Feature matching gives precise geometry, but a limited keypoint budget can be spent on the
background. Segmentation captures target shape, but its corners have ambiguous ordering. We present
FlatSAM, which combines an independently maintained keypoint tracker with a quadrilateral fitted to
a shared SAM3 mask. It prioritises keypoints inside the target, estimates pose directly between the
initial and current frames, and selects between keypoint and mask estimates using match support.
The same matches assign the mask corners, without an additional appearance model. FlatSAM achieves
88.0 p@15 on re-annotated PlanarTrack and 97.23 p@15 on POT-210. It leads on all five benchmarks.
These results show how target-focused matching and mask-based geometry complement each other for
planar tracking.

## Results
| Method | Segmentation model | Geometry branch | POT-210 p@5 | POT-210 p@15 | PlanarTrack p@5 | PlanarTrack p@15 |
|---|---|---|---:|---:|---:|---:|
| SIFT | N/A | keypoints | 65.8 | 69.6 | 18.8 | 26.8 |
| LISRD | N/A | keypoints | 68.3 | 79.2 | 23.5 | 37.3 |
| HDN | N/A | regressed | 70.9 | 92.4 | 31.9 | 58.0 |
| HVC-Net | N/A | regressed | 91.4 | 95.8 | 48.8 | 64.3 |
| WOFT | N/A | flow | 90.0 | 95.4 | 48.9 | 65.9 |
| SAM-H | SAM2 | Hough | 68.6 | 89.4 | 62.4 | 80.5 |
| SAM-H, segmenter control | SAM3 | Hough | 70.9 | 90.3 | <ins>65.0</ins> | <ins>82.4</ins> |
| WOFTSAM | SAM2 | flow+Hough | 91.9 | <ins>97.5</ins> | 57.4 | 77.6 |
| WOFTSAM, segmenter control | SAM3 | flow+Hough | 92.1 | 97.2 | 58.3 | 79.2 |
| keypoint branch only (ours) | SAM3 | keypoints | <ins>92.4</ins> | 97.3 | 56.7 | 78.2 |
| **FlatSAM (ours)** | SAM3 | keypoints+Hough | **93.4** | **98.5** | **65.6** | **88.0** |

Pooled p@k (%) on the 1,004 changed-frame POT-210 protocol and re-annotated PlanarTrack. POT-210
values follow WOFTSAM; PlanarTrack uses WOFTSAM's updated initial poses. SIFT, LISRD, and HDN are
rescored released predictions (not reruns), with fallback on 236 degenerate frames. POT-210 uses
WOFT's changed-frame labels; our PlanarTrack rows are evaluated predictions. SAM3 controls change
only the segmenter.

## Installation
```bash
cd flatsam-demo

./setup.sh

source .envrc
```

`setup.sh` needs a Hugging Face token with access to the gated [SAM3 weights](https://huggingface.co/facebook/sam3).
Request access there, create a Read token at <https://huggingface.co/settings/tokens>, and paste it
when `setup.sh` asks (or set `HF_TOKEN` in `.env`). Settings (dataset and output paths) live in
`.env`, created from `.env.example` on the first run.

Tested on Linux x86_64 with Python 3.12 and CUDA 12.6, on NVIDIA A100 and RTX 4000 SFF Ada (20 GB) GPUs.

## Evaluation
The scripts below reproduce the **FlatSAM (ours)** row of the results table.

### 1. Put the datasets in place
By default the datasets are expected in `data/` inside the repository. To keep them elsewhere,
set `POT_BASE`, `PT_BASE` and `PT_INITFIX_DIR` in `.env`, or pass `--base-dir` (and
`--initfix-dir` for PlanarTrack) to `run.py` and `evaluate.py`.

```
data
 - POT-210                                  (POT_BASE)
   - annotation
     - WOFT_reannotation_coarse_onlychanged
       - V01_1_gt_points.txt
       - V01_1_flag.txt
       - ...
   - extracted
     - images
       - V01_1
         - 00000001.jpg
         - ...
       - ...
 - PlanarTrack
   - PlanarTrack-Data                       (PT_BASE)
     - annos
       - Seq_00001.txt
       - Seq_00001_flag.txt
       - ...
     - sequences
       - Seq_00001
         - 00001.jpg
         - ...
     - training-test-splits
       - test_seq.txt
   - PlanarTrack_init_reannotation          (PT_INITFIX_DIR)
     - Seq_00003.txt
     - ...
```

**POT-210.** Download the videos from the [POT-280 website](https://liangp.github.io/data/pot280/)
(only objects V01–V30 form POT-210) and convert them to frames:
```bash
# inside data/POT-210/, next to the V01/ ... V30/ folders with the .avi videos
for o in $(seq -w 1 30); do for t in $(seq 7); do s=V${o}_$t
  mkdir -p extracted/images/$s
  ffmpeg -i V$o/$s.avi -q:v 1 -vsync 0 -f image2 extracted/images/$s/%08d.jpg
done; done
```
The ground truth is the WOFT re-annotation:
```bash
wget https://cmp.felk.cvut.cz/~serycjon/WOFT/results/POT-210-reannot_coarse_onlychanged.tar.gz
mkdir -p data/POT-210/annotation
tar xzf POT-210-reannot_coarse_onlychanged.tar.gz -C data/POT-210/annotation
```

**PlanarTrack.** Download the dataset from the
[PlanarTrack website](https://hengfan2010.github.io/projects/PlanarTrack/). The re-annotated
initial corners of the 300 test sequences are released with WOFTSAM:
```bash
git clone --depth 1 https://github.com/serycjon/WOFTSAM
cp -r WOFTSAM/PlanarTrack_init_reannotation data/PlanarTrack/
```

### 2. Track and evaluate
```bash
CFG=configs/tracker/flatsam_final.py

# POT-210 (210 sequences)
python run.py pot --config $CFG
python evaluate.py pot --config $CFG

# PlanarTrack test split with the re-annotated initial corners (300 sequences)
python run.py pti --split test --config $CFG
python evaluate.py pti --split test --config $CFG
```
`run.py` writes one `<sequence>.txt` per sequence (4 corners per frame, `x1 y1 ... x4 y4`) to
`results/flatsam_final/POT-210/` (or `.../PlanarTrackInitfix/`) and skips sequences that are already done, so an
interrupted run can be restarted. `evaluate.py` prints the pooled P@5 and P@15 and writes
per-sequence scores to `eval_results.json` in the same folder.

Expected output: P@5 93.4 / P@15 98.5 on POT-210 and P@5 65.6 / P@15 88.0 on PlanarTrack.
Results can differ slightly between GPU models. A full POT-210 run takes about 18 hours on an
RTX 4000 SFF Ada. For a quick check, run a few sequences and pass the same `--sequences` to
both scripts:
```bash
python run.py pot --config $CFG --sequences V01_7 V12_1
python evaluate.py pot --config $CFG --sequences V01_7 V12_1
```

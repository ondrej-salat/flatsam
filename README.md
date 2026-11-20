# FlatSAM: Segmentation-Focused Correspondences for Precise and Robust Planar Tracking

## Abstract

Planar tracking estimates a homography mapping a target region to its current location.
Texture-based matching is precise but becomes unreliable when the target is small, blurred, or weakly
textured. A segmentation tracker is much more robust, but its output, the mask, does not fully specify a homography.
We propose FlatSAM, which uses the keypoint pose wherever this is reliable, falling back to the pose estimated from the
mask contour otherwise. The segmentation mask also focuses the keypoint budget. To achive maximum accuracy, FlatSAM
rectifies each frame with its own first-pass pose estimate and runs again to remove residual misalignments.
FlatSAM sets a new state-of-the-art: it has the highest p@15 and p@5 on six standard public benchmarks.
It reaches 66.1 p@5 and 87.7 p@15 on the PlanarTrack test split and 84.5 and 97.3 on POT-210, and removes 38\,\% and
28\,\% of the p@15 errors of the best previous tracker on PlanarTrack and MPOT-3K, the two benchmarks far from
saturation.
Ablations show that the mask pose complements correspondence geometry, especially on PlanarTrack, and that the
refinement pass adds 0.8 points on PlanarTrack, 0.8 on MPOT-3K and 1.7 on UCSB.

## Results

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">method</th>
      <th rowspan="2" align="left">metric</th>
      <th colspan="2"><a href="https://liangp.github.io/data/pot280/">POT-210</a></th>
      <th rowspan="2"><a href="https://liangp.github.io/data/pot280/">POT-280</a></th>
      <th rowspan="2"><a href="https://hengfan2010.github.io/projects/PlanarTrack/">PT<sub>TST</sub></a></th>
      <th rowspan="2"><a href="https://github.com/LinChen-Cherry/POIC">POIC</a></th>
      <th rowspan="2">UCSB</th>
      <th rowspan="2"><a href="https://github.com/nku-zhichengzhang/MPOT">MPOT-3K</a></th>
    </tr>
    <tr><th>all</th><th>re-ann.</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2"><b>FlatSAM (ours)</b></td><td>p@5</td><td align="center"><b>84.5</b></td><td align="center"><b>94.4</b></td><td align="center"><b>80.5</b></td><td align="center"><b>66.1</b></td><td align="center"><b>98.6</b></td><td align="center"><b>98.0</b></td><td align="center"><b>70.6</b></td></tr>
    <tr><td>p@15</td><td align="center"><b>97.3</b></td><td align="center"><b>98.3</b></td><td align="center"><b>96.9</b></td><td align="center"><b>87.7</b></td><td align="center"><b>99.7</b></td><td align="center"><b>99.3</b></td><td align="center"><b>93.4</b></td></tr>
    <tr><td rowspan="2">WOFTSAM</td><td>p@5</td><td align="center"><ins>82.7</ins></td><td align="center"><ins>91.9</ins></td><td align="center"><ins>78.5</ins></td><td align="center">57.4</td><td align="center"><ins>97.6</ins></td><td align="center"><ins>97.9</ins></td><td align="center"><ins>67.4</ins></td></tr>
    <tr><td>p@15</td><td align="center"><ins>95.7</ins></td><td align="center"><ins>97.5</ins></td><td align="center"><ins>95.0</ins></td><td align="center">77.6</td><td align="center"><ins>99.2</ins></td><td align="center"><ins>98.8</ins></td><td align="center"><ins>90.8</ins></td></tr>
    <tr><td rowspan="2">SAM-H</td><td>p@5</td><td align="center">64.4</td><td align="center">68.6</td><td align="center">64.1</td><td align="center"><ins>62.0</ins></td><td align="center">68.4</td><td align="center">15.3</td><td align="center">54.7</td></tr>
    <tr><td>p@15</td><td align="center">88.7</td><td align="center">89.4</td><td align="center">88.5</td><td align="center"><ins>80.1</ins></td><td align="center">80.8</td><td align="center">64.9</td><td align="center">82.0</td></tr>
  </tbody>
</table>

Accuracy on six standard benchmarks measured by frame-pooled p@5 and p@15 (%). POT-210 is scored on
every annotated frame with the original annotation (all) and on the 1,004 frames re-annotated by WOFT
(re-ann.). PlanarTrack (PT<sub>TST</sub>, test split) uses the re-annotated starting corners released with
WOFTSAM for every method; its SAM-H scores are the published ones (Table 1 of the WOFTSAM paper). Other
SAM-H entries are our rescoring of the stored run. UCSB uses the original ground truth mapped to the raw
frames with the calibration published with the dataset. Best results in bold, second best underlined.

## Viewer

`FlatSAM_demo.zip` contains a self-contained browser viewer (`viewer_demo/index.html`) with 12 sequences of the six
benchmarks, the sequences of the paper's figures. Frame by frame it shows the ground truth, the
SAM3 mask, the first and the refinement pass of FlatSAM with their keypoint and mask branches, and WOFTSAM and SAM-H.

```bash
unzip FlatSAM_demo.zip
python3 viewer_demo/start_demo.py   # serves the folder on 127.0.0.1 and opens the viewer
```

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

`third_party/sam3` is a modified copy of [SAM3](https://github.com/facebookresearch/sam3) (streaming
tracking interface added), distributed under the SAM License in `third_party/sam3/LICENSE`.

Tested on Linux x86_64 with Python 3.12 and CUDA 12.6, on NVIDIA A100 and RTX 4000 SFF Ada (20 GB) GPUs.

## Evaluation

Each benchmark takes three commands, which reproduce the FlatSAM row of the results table:

```bash
python run.py <dataset> --save-masks              # first pass
python refine.py <dataset>                        # refinement pass on every frame rectified with its first-pass pose
python evaluate.py <dataset> --results-name warp  # p@5 and p@15 of the refinement pass, FlatSAM's output
```

Without `--config`, `run.py` uses the final method (`configs/tracker/flatsam_final.py`, `DEMO_CONFIG` in `.env`).
Datasets go to `data/` by default (the `*_BASE` paths in `.env`); `--base-dir` overrides it per run.

### 1. Datasets

**[POT-210](https://liangp.github.io/data/pot280/)** (objects V01–V30 of POT-280):

```bash
for o in $(seq -w 1 30); do python scripts/download_pot280.py --root data/POT-210 --object V$o; done
wget https://cmp.felk.cvut.cz/~serycjon/WOFT/results/POT-210-reannot_coarse_onlychanged.tar.gz
tar xzf POT-210-reannot_coarse_onlychanged.tar.gz -C data/POT-210/annotation

python run.py pot --save-masks
python refine.py pot
python evaluate.py pot --results-name warp                        # re-ann. column
python evaluate.py pot --results-name warp --original-annotation  # all column
```

**[PlanarTrack](https://hengfan2010.github.io/projects/PlanarTrack/)**: download and unpack to
`data/PlanarTrack/PlanarTrack-Data/` (`annos/`, `sequences/`, `training-test-splits/`), then add the re-annotated
initial corners:

```bash
git clone --depth 1 https://github.com/serycjon/WOFTSAM
cp -r WOFTSAM/PlanarTrack_init_reannotation data/PlanarTrack/

python run.py pti --split test --save-masks
python refine.py pti
python evaluate.py pti --split test --results-name warp
```

**[POT-280](https://liangp.github.io/data/pot280/)**:

```bash
python scripts/download_pot280.py --root data/POT-280 --object all

python run.py pot280 --save-masks
python refine.py pot280
python evaluate.py pot280 --results-name warp
```

**[POIC](https://github.com/LinChen-Cherry/POIC)**:

```bash
git clone --depth 1 https://github.com/LinChen-Cherry/POIC data/POIC

python run.py poic --save-masks
python refine.py poic
python evaluate.py poic --results-name warp
```

**UCSB:** Unpack to `data/UCSB/<sequence>/`
(e.g. `data/UCSB/bricks_motion1/frame00001.jpg`), then add the corrected ground truth from
[`ucsb_mapped_gt.zip`](ucsb_mapped_gt.zip) in this repository:

```bash
unzip -o -j ucsb_mapped_gt.zip 'ucsb_mapped_gt/*.txt' -d data/UCSB

python run.py ucsb --save-masks
python refine.py ucsb
python evaluate.py ucsb --results-name warp
```

**[MPOT-3K](https://github.com/nku-zhichengzhang/MPOT#mpot-3k-dataset)**: download and unpack to
`data/MPOT-3K/` (`list/`, `test/`, ...), then:

```bash
python run.py mpot --save-masks
python refine.py mpot
python evaluate.py mpot --results-name warp
```

## Your own video

`demo.py` tracks a planar object in any video and writes a copy with the tracked quadrilateral drawn in:

```bash
python demo.py video clip.mp4 --corners x1 y1 x2 y2 x3 y3 x4 y4   # writes clip_tracked.mp4
```

`--corners` are the four corners of the target in the first frame, in pixels, clockwise from the top-left.

The demo runs the first pass only, without the refinement pass. For FlatSAM's full two-pass output, as reported in
the paper, use `run.py` and `refine.py` as in [Evaluation](#evaluation).

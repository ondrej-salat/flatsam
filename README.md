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
      <th rowspan="2"><a href="https://web.archive.org/web/20160608015230/http://ilab.cs.ucsb.edu/tracking_dataset_ijcv/">UCSB</a></th>
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

The scripts below reproduce the FlatSAM row of the results table: `run.py` runs the first pass and
`subset_viewer/rerun/run_warp_pipeline.py` (the script of the paper's runs) the refinement pass on every frame
rectified with its first-pass pose. Without `--config`, every script uses the final method
(`configs/tracker/flatsam_final.py`, `DEMO_CONFIG` in `.env`).

### 1. Get a dataset, track and evaluate

Datasets go to `data/` by default (the `*_BASE` paths in `.env`); `--base-dir` overrides it per run.

```bash
# refinement pass on every sequence of a finished first pass: refine <dataset key> <dataset folder>
refine() {
  { echo dataset,dataset_key,sequence,tier,base_dir
    for f in results/flatsam_final/$2/*_debug.npz; do echo "$2,$1,$(basename "$f" _debug.npz),,"; done
  } > results/tasks_$2.csv
  n=$(( $(wc -l < results/tasks_$2.csv) - 1 ))
  for ((i = 0; i < n; i++)); do
    python subset_viewer/rerun/run_warp_pipeline.py --tasks results/tasks_$2.csv --task-index $i \
      --outroot results/warp --mode canvas --max-scale 1 --first-pass-dir results/flatsam_final
  done
}
```

**[POT-210](https://liangp.github.io/data/pot280/)** (objects V01–V30 of POT-280):

```bash
for o in $(seq -w 1 30); do python scripts/download_pot280.py --root data/POT-210 --object V$o; done
wget https://cmp.felk.cvut.cz/~serycjon/WOFT/results/POT-210-reannot_coarse_onlychanged.tar.gz
tar xzf POT-210-reannot_coarse_onlychanged.tar.gz -C data/POT-210/annotation

python run.py pot --save-masks
refine pot POT-210
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
refine planartrack-initfix PlanarTrackInitfix
python evaluate.py pti --split test --results-name warp
```

**[POT-280](https://liangp.github.io/data/pot280/)**:

```bash
python scripts/download_pot280.py --root data/POT-280 --object all

python run.py pot280 --save-masks
refine pot280 POT-280
python evaluate.py pot280 --results-name warp
```

**[POIC](https://github.com/LinChen-Cherry/POIC)**:

```bash
git clone --depth 1 https://github.com/LinChen-Cherry/POIC data/POIC

python run.py poic --save-masks
refine poic POIC
python evaluate.py poic --results-name warp
```

**[UCSB](https://web.archive.org/web/20160608015230/http://ilab.cs.ucsb.edu/tracking_dataset_ijcv/)**: the frames are
no longer publicly available; they can be obtained from the dataset's authors. Unpack them to `data/UCSB/<sequence>/`
(e.g. `data/UCSB/bricks_motion1/frame00001.jpg`), then add the corrected ground truth from
[`ucsb_mapped_gt.zip`](ucsb_mapped_gt.zip) in this repository:

```bash
unzip -o -j ucsb_mapped_gt.zip 'ucsb_mapped_gt/*.txt' -d data/UCSB

python run.py ucsb --save-masks
refine ucsb UCSB
python evaluate.py ucsb --results-name warp
```

**[MPOT-3K](https://github.com/nku-zhichengzhang/MPOT#mpot-3k-dataset)** (on request, see the link): download and unpack to
`data/MPOT-3K/` (`list/`, `test/`, ...), then:

```bash
python run.py mpot --save-masks
refine mpot MPOT-3K
python evaluate.py mpot --results-name warp
```

### 2. Output

`run.py` writes the first pass to `results/flatsam_final/<dataset>/`: one `<sequence>.txt` per sequence
(4 corners per frame, `x1 y1 ... x4 y4`), `<sequence>_debug.npz` and, with `--save-masks`, the SAM3 masks the
refinement pass reads. `refine` writes FlatSAM's output to `results/warp/<dataset>/<sequence>.txt` (same format).
Both skip sequences that are already done, so an interrupted run can be restarted. `evaluate.py --results-name warp`
prints the pooled P@5 and P@15 of the output (the FlatSAM row of the table above) and writes per-sequence scores to
`eval_results.json` (`eval_results_all.json` with `--original-annotation`) in the same folder; without
`--results-name` it scores the first pass, whose p@15 is (Table 4 of the paper):

| POT-210 all | POT-210 re-ann. | POT-280 | PT<sub>TST</sub> | POIC | UCSB | MPOT-3K |
|---:|---:|---:|---:|---:|---:|---:|
| 97.1 | 98.3 | 96.7 | 86.9 | 99.3 | 97.5 | 92.6 |

Results can differ slightly between GPU models. On an RTX 4000 SFF Ada a full POT-210 run takes about
18 hours for the first pass and about as long again for the refinement pass. For a quick check, run a few sequences:

```bash
python run.py pot --sequences V01_7 V12_1 --save-masks
refine pot POT-210
python evaluate.py pot --results-name warp --sequences V01_7 V12_1
```

### 3. Notes

- **No training.** FlatSAM uses the pretrained SAM3 and LoMaR models as they are; nothing is trained or fine-tuned.
- **Metric.** The alignment error of a frame is the root-mean-square distance of the four corners from the ground
  truth; p@k is the percentage of annotated frames (frame 0 excluded) with error ≤ k px, pooled over all frames of a
  benchmark (Eq. 3–4 of the paper).
- **Runs.** Each result is a single run per benchmark with fixed random seeds (`--seed 0`, the default). CUDA kernels
  are not forced to be deterministic, so results can differ slightly between runs and GPU models.
- **Not included.** The ablations (Tables 2–4: single branches, branch oracle, corner order, threshold sweep) and the
  bootstrap intervals (Supplementary Sec. E) were computed from stored runs and are not part of this demo.

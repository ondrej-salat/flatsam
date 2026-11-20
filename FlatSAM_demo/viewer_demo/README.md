# FlatSAM viewer demo

A small, self-contained browser viewer for 12 example sequences of POT-210, POT-280, PlanarTrack, POIC and UCSB. It needs no internet access.

## Start

- Double-click `start_demo.command` (macOS) or run `python3 start_demo.py` in this folder. It serves the folder on localhost and opens the viewer. Python 3 standard library only.
- Opening `index.html` directly also works for looking at the sequences, but a page opened from a file cannot read its own images, so **Screenshot** and **Record** only work through `start_demo.py`.

**Record** (MP4) needs a secure page: browsers only offer the video encoder on https or on localhost, so use `start_demo.py` (it serves http://127.0.0.1) and not a plain http address of another machine. It works in current Chrome, Edge, Firefox and Safari (tested with Chrome, Firefox and the WebKit engine of Safari; not with Edge itself).

`flatsam_viewer_demo.mp4` (next to this folder) is a 30 s screen recording of the viewer.

## The sequences

The sequences are those of the figures of the paper, where the tracker performs well over the whole sequence: first the figures of the main paper (Figure 1 and the PlanarTrack examples), then Figure S3 of the supplementary (the comparison of our variants and the baselines) and the rectification figure. A sequence opens at the frame shown in the paper ("Paper frame" button). The set is a favorable sample, like Figure S3 itself: the frames of the paper figures are chosen where both of our passes are accurate, and on most frames of the benchmarks all methods are accurate (see the supplementary for how often we or a baseline win).

| Sequence | Where in the paper | Frame |
|---|---|---|
| PlanarTrack Seq_00782 | PlanarTrack examples figure (card behind an acrylic stand) | 397 |
| PlanarTrack Seq_00434 | PlanarTrack examples figure | 499 |
| POT-210 V28_2 | rotation figure | 416 |
| POT-210 V04_4 | intro figure (POT-210 examples) | 353 |
| PlanarTrack Seq_00959 | Figure 1 (screen with changing content) | 241 |
| POT-210 V13_7 | Figure S3 (unconstrained) | 203 |
| POT-280 V39_5 | Figure S3 (occlusion) | 495 |
| PlanarTrack Seq_00916 | Figure S3 (motion blur) | 356 |
| PlanarTrack Seq_00662 | Figure S3 (occlusion) | 340 |
| POIC box2 | Figure S3 (illumination change) | 536 |
| UCSB bricks_unconstrained | Figure S3 | 88 |
| PlanarTrack Seq_00952 | rectification and corner-order figures | 527 |

The paper figures show one frame per sequence. The viewer shows every frame, so it also shows frames where a method fails; the list gives the scores of the whole sequence and is sorted by Δ, the margin of FlatSAM over the better of the two baselines, highest first.

## What you see

Four windows show the current frame, in sync with the slider and the graphs: the first pass FlatSAM with its keypoint and mask branch (top left), the ground truth and the SAM3 mask (top right), the refinement pass FlatSAM with its branches (bottom left), and WOFTSAM and SAM-H (bottom right). The warp overlay is in the top right window: the first frame warped onto the current frame by the first pass (green, half transparent), the pose with which the refinement pass rectifies the frame; where the warped first frame matches the target and the ground truth, the first pass is right.

**Baselines and scores.** WOFTSAM and SAM-H are the SAM 2 versions of the main table of the paper. The pooled p@5 and p@15 over a whole benchmark equal the paper's (POT-210 with the original annotation of every annotated frame), up to 0.1 points.

**Layers.** The layers of a window are listed next to it, with their color and line style and their error and IoU in the current frame. Every layer can be hidden (checkbox); all are shown at the start. The warp overlay has its switch under "Ground truth". Ground truth is solid green, the two passes are solid, the branches are dashed, WOFTSAM and SAM-H are dotted. Each pass has its own color family (the first-pass branches use the colors of the overview figure, Fig. 2).

**Graphs.** The three graphs below the windows show the IoU with the ground truth over time: (1) the first pass and its two branches, (2) the refinement pass and its two branches, (3) FlatSAM against WOFTSAM and SAM-H. The line under a graph describes it: the strip shows which branch the pose gate chose in each frame, grey areas have no ground truth, and under the first two graphs the number of keypoint inliers of the current frame is given ("keypoint inliers = 46"; the gate threshold is T = 35, the minimum 8). Click or drag in a graph to move through the sequence.

**Sliding panels.** The edge of the sequence list and the bar above the graphs work the same way: drag to change the size; drag to the edge of the screen to hide the panel; a small tab stays at the edge, click it to show the panel again (double-click: default size). The windows use the freed space.

**List.** The list on the left shows, per sequence, P@k of FlatSAM (the refinement pass), WOFTSAM and SAM-H, and the difference Δ between FlatSAM and the better of the two baselines; k is 5, 10 or 15 pixels, chosen above the list (P@15 is the share of frames with a corner error of at most 15 pixels, as in the paper). It is sorted by Δ, highest first; click a column heading to sort by it. The star in front of a sequence (or key F) marks it; "Starred" in the dataset selector shows only those, and "Copy starred" copies their names. Stars are kept in this browser only.

**Toolbar** above the windows: **+**, **-**, **Reset** zoom all windows (the mouse does not zoom: dragging or scrolling moves the zoomed windows, double-click resets the zoom); **Screenshot** saves a PNG of the four windows at the current frame with the numbers of that frame (also stored in the PNG metadata); **Record** works with Play: click Record (it shows "press Play"), press Play, and the frames shown while playing are recorded as an MP4 of the four windows, with the frame number in the top right; Pause, the end of the sequence or jumping in the timeline saves the file (click Record again before pressing Play to cancel); **Paper frame** jumps to the frame shown in the paper.

**Keys:** Left / Right: previous / next frame (Shift: 10 frames); Space: play; Up / Down: previous / next sequence of the list; F: star; P: screenshot; 0: reset the zoom; ?: help.

**The address bar holds a link to the current view**: sequence, frame, hidden layers, the warp overlay switch, the hidden panels and the zoom. Copy it to send exactly what you see (the demo folder is needed on the other side, served by `start_demo.py` or opened as a file).

Scores are computed on the original frames; the stored frames are downscaled to 416 px width for the demo. Portrait PlanarTrack sequences are shown turned by 90 degrees (tag "rotated"); the scores are unchanged. WOFTSAM and SAM-H on UCSB use the corrected annotations, like all other methods in the paper.

## Add more sequences

`tools/add_sequence.py` adds a sequence (or replaces it) in this folder. It needs numpy and opencv: `pip install numpy opencv-python-headless`.

1. Track the sequence with the code of this repository (`run.py`, see the main README). With the config `flatsam_final`, `run.py` writes the **first pass** as one poses file per sequence, `results/flatsam_final/<DATASET>/<sequence>.txt`: one row per frame with eight numbers `x1 y1 x2 y2 x3 y3 x4 y4` (the four corners in the order of the annotation). Row 0 is the initialisation. Rows of NaN mean "no pose". The poses file of the refinement pass has the same format and goes to `--refinement`.
2. Add it:

   ```
   python3 tools/add_sequence.py --dataset MyData --sequence clip1 \
       --frames path/to/frames_folder \
       --first-pass results/flatsam_final/MyData/clip1.txt
   ```

   Optional files in the same format: `--refinement`, `--debug` (see below), `--gt` (enables the scores; rows of NaN or of all zeros, as in some annotation files, count as "no annotation" and are not scored), `--woftsam`, `--samh`. Instead of `--frames` you can give `--video clip.mp4`. Layers you do not give are shown as unavailable. `--width 416 --quality 46` control the size of the stored frames.
3. Reload the page. The sequence shows up in the list under its dataset name (do not use the name "PlanarTrack", the page hides it).

Without `--debug` the branch layers and the SAM3 mask stay empty for an added sequence (and the first graph shows only the result of the first pass). `run.py` writes `<sequence>_debug.npz` next to the first-pass poses; pass it as `--debug results/flatsam_final/MyData/clip1_debug.npz` and the keypoint and mask branch layers of the first pass, the chosen branch and the inlier count appear. The refinement-pass branches and the SAM3 mask are not read. The folder grows by about 20 KB per frame at the default settings.

## Folder layout

`index.html` is the viewer (one file). `data/manifest.js` lists the sequences, `data/<dataset>/<sequence>.js` holds the outlines and scores of one sequence, `frames/<dataset>/<sequence>/NNNNN.jpg` its frames. `vendor/mp4-muxer.js` (MIT, see `vendor/mp4-muxer.LICENSE`) writes the MP4 files.

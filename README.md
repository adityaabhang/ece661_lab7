# ECE 661 HW7: Otsu, morphology, distance maps, watershed, color

## Files
- `hw7_AdityaAbhang.py`: all the code (Tasks 1 to 5), one function per thing the tasks ask for.
- `hw7_images/`: the supplied data (`coins.png`, `coins_truth.png`, `broken_rectangle1.jpg`, `discs_rgb.png`, `coins_rgb.png`). It must sit next to the `.py` file.
- `output/`: created by the code; holds every figure it makes.

## How to run
- Install the packages (versions below): `pip install numpy scipy pillow matplotlib Watershed==2.2.3`
- Put `hw7_AdityaAbhang.py` and the `hw7_images/` folder in the same directory, and run from that directory.
- Run everything: `python hw7_AdityaAbhang.py`
- No display is needed. If your machine tries to open a window, run: `MPLBACKEND=Agg python hw7_AdityaAbhang.py`
- The whole run takes about a minute (Task 4 is most of it).
- The file is safe to import: the work is inside functions, and `main()` runs only under `if __name__ == '__main__':`.
- `main()` first runs `check_setup()`, which checks the data and the module and prints the versions. If it fails, the run stops.
- To run one task only: `python -c "import hw7_AdityaAbhang as h; h.task3()"` (use `task1` to `task5`).
- Paths are set in variables at the top of the file (`DATA_DIR`, `OUT_DIR`).

## What the run prints and saves
- Task 1: k*, the class statistics, eta, the check of equation (16), and the table at k*, 60, 140 and 180.
- Task 2: the structuring element tables, the openings, and the closings.
- Task 3: the diamond arrays, the distance map checks against SciPy, the marker table, and the 15 chosen markers.
- Task 4: the influence zones, the automatic watershed runs, the marker discs runs, the hybrid, and the seven-row scoreboard.
- Task 5(a): k*, area and components of each mask, and the patch means.
- Figures go to `output/`: `task1a_otsu.png`, `task2b_best_opening.png`, `task2c_closing.png`, `task3_distance_markers.png`, `task4a_influence_zones.png`, `task4b_default.png`, `task4b_best.png`, `task4c_best.png`, `task4d_hybrid.png`, `task5a_channels.png`.

## Not code
- Task 5(b) is done by hand in the browser (SAM 3 Playground, https://aidemos.meta.com/segment-anything/) on `coins_rgb.png`, with the prompts `circle`, `silver circle` and `gold circle`. The screenshots are in the report.
- The hand-worked arrays of Task 3(a) and the sketch of Task 4(a) are done on paper.

## Versions it was run on
- Python 3.11.16
- NumPy 2.4.6
- SciPy 1.17.1
- Pillow 11.1.0
- Matplotlib 3.10.0
- Watershed 2.2.3

## AI assistance
- The functions that an AI assistant (Claude) wrote entirely are marked in their docstrings with "Generated entirely by an AI assistant (Claude)".

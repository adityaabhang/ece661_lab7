#!/usr/bin/env python

##  hw7_skeleton.py

__version__   = '1.0.0'
__author__    = "[Aditya Uday Abhang] (aabhang@purdue.edu)"
__date__      = '2026-October-21'


__doc__ = '''

    hw7_skeleton.py

    This is the skeleton for Homework 7 of ECE 661, Computer Vision, Fall 2026.

    Rename this file to hw7_<FirstName><LastName>.py before you submit it.

    Fifteen coins have been dumped out of a piggy bank onto a table, together
    with a little dust: coins.png. Some of the coins touch.  
    Your job is to count them and to outline every one,
    with nothing but Otsu's threshold, the morphology of Lecture 14 and the
    watershed of Lecture 15, all of it run through Prof. Kak's Watershed
    module.  Each task is one more step along the way, and each ends with a
    number that tells you how far along you are.

        Task 1    a threshold nobody has to guess: Otsu's method (Otsu 1979)
        Task 2    erosion, dilation, opening and closing, as the module does
                  them (Lecture 14)
        Task 3    distance maps and the MAT (Rosenfeld and Kak, pages 209-216),
                  and from them one marker per coin
        Task 4    the watershed (Lecture 15): influence zones, the automatic
                  watershed, marker control, and a hybrid of the two
        Task 5    color: Otsu's threshold on the channels of a color picture,
                  and a pretrained segmenter (SAM 3), which you use in a
                  browser

    What you may and may not use:

        everywhere     numpy, scipy and matplotlib.  Prof. Kak's Watershed
                       module (pip install Watershed) does the morphology and
                       the watershed, and the "Given helpers" below call it
                       for you.  No cv2 and no skimage: do not import either.
                       Pillow loads the pictures.
        Task 1         your own Otsu.  No library threshold routine.
        Task 2         erosion and dilation go through the module's erode() and
                       dilate(), via module_erode() and module_dilate() below.
                       Not scipy.ndimage.binary_erosion: it uses other
                       conventions, and the conventions are the point of
                       Task 2(a).
        Task 3         distance_two_pass() is yours.  You may call
                       scipy.ndimage.distance_transform_cdt to CHECK it, not
                       to produce the numbers you report.  scipy.ndimage.
                       maximum_filter is permitted for the local-maximum test.
        Task 4         the watershed is the module's, through run_watershed()
                       and influence_zones().
        Task 5         part (a) is your own Otsu on each channel.  SAM 3 is
                       for part (b) only.  It is a web page, so there is no
                       code for it, and no pretrained model stands in for any
                       of Tasks 1 to 4.

    Report the real-valued quantities of Task 1 to four significant figures,
    the Dice of each coin to two decimals and every mean Dice to three.  A
    Dice of 0.8 for coin 6 hides that it is 0.78, which is under the line that
    decides whether a coin is found.

    Everything in the section headed "Given helpers" below is already written.
    Run check_setup() before anything else: it says whether
    the data and Prof. Kak's module are where this file expects them.

    Note that the work is all inside functions and that the calls at the
    bottom sit behind the usual __main__ guard.  Keep it that way.  

    Fill in every place marked TODO, and delete the NotImplementedError when
    you do.

    Usage:      python hw7_skeleton.py
    Requires:   numpy, scipy, matplotlib, Pillow, Watershed   (version 2.2.3
                or later).  No display or Tk is needed: the helpers below
                switch off every window the module would otherwise open.

    Course policy on AI assistants applies to this file.  Your report must
    open with an "AI Usage: XX%" line, and any function here that was
    generated entirely by an AI assistant must say so in its own docstring.
'''

import os
import io
import sys
import time
import types
import shutil
import tempfile
import contextlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


##  --------------------------------------------------------------------------
##  Paths and constants.  These live here, at the top, not buried inside a
##  function:  a hard coded C:\Users\... path cannot be run by anyone else,
##  your grader included.
##  --------------------------------------------------------------------------

DATA_DIR  = "hw7_images"
OUT_DIR   = "output"

COINS     = "coins.png"                  # 128 x 128, 8-bit gray
TRUTH     = "coins_truth.png"            # 0 = table, 1 ... 15 = the coins
RECT      = "broken_rectangle1.jpg"      # from the module's Examples directory
DISCS     = "discs_rgb.png"              # 128 x 128, RGB: Task 5(a)
COINS_RGB = "coins_rgb.png"              # 512 x 512, RGB: Task 5(b) upload

MIN_AREA  = 45      # a region of fewer pixels than this is dust, not a coin
N_COINS   = 15      # what the answer should be

##  The seven-by-seven diamond of Task 3(a).  Rosenfeld and Kak assume that
##  the picture is surrounded by 0's (page 214), so imagine a border of 0's.
DIAMOND = np.array([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0]], dtype=bool)


def load_gray(name=COINS):
    """The shipped picture as a 2-D uint8 array (row, column)."""
    return np.array(Image.open(os.path.join(DATA_DIR, name)).convert("L"))


def load_truth():
    """coins_truth.png as an int array: 0 = table, k = the k-th coin.

    Use it to MEASURE (through evaluate() below), never to decide anything.
    """
    return np.array(Image.open(os.path.join(DATA_DIR, TRUTH))).astype(int)


def load_binary(name=RECT):
    """A shipped black-and-white picture as a boolean array.  The .jpg files
    are not exactly 0 and 255 after compression, so they are cut at 128."""
    gray = Image.open(os.path.join(DATA_DIR, name)).convert("L")
    return np.array(gray) >= 128


def load_rgb(name=DISCS):
    """A shipped color picture as an (H, W, 3) uint8 array: R, G, B."""
    return np.array(Image.open(os.path.join(DATA_DIR, name)).convert("RGB"))


def rgb_to_gray(rgb):
    """The gray level of a color picture, 0.299 R + 0.587 G + 0.114 B, rounded
    to the nearest integer (the weights of Pillow's convert("L"))."""
    weights = np.array([0.299, 0.587, 0.114])
    return np.rint(np.asarray(rgb, float) @ weights).astype(np.uint8)


def save_figure(fig, name):
    """Write a figure into OUT_DIR, creating it if it does not exist."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")


##  ============================================================================
##  Given helpers.  Please do not change them.
##  ============================================================================

_QUIET = None


def _quiet_watershed():
    """Prof. Kak's Watershed class with every pop-up window switched off.
    Nothing else about it is changed.

    The module imports tkinter and PIL.ImageTk when it is imported, only so
    that it can open display windows.  If either is missing on your machine
    (no Tk, or a headless server) we plug in empty stand-ins so that the
    import still succeeds.  The class is built on the first call, so that this
    file can be imported on a machine that does not have the module."""
    global _QUIET
    if _QUIET is not None:
        return _QUIET
    try:
        import tkinter                                   
        from PIL import ImageTk                          
    except Exception:
        class _Dummy:
            def __init__(self, *a, **k): pass
            def winfo_screenwidth(self): return 1920
            def winfo_screenheight(self): return 1080
            def __getattr__(self, name): return lambda *a, **k: None
        tk = types.ModuleType("tkinter")
        tk.Tk = tk.Frame = tk.Label = tk.Canvas = tk.Button = _Dummy
        tk.Toplevel = tk.PhotoImage = _Dummy
        tk.TclError = Exception
        tk.mainloop = lambda *a, **k: None
        tkc = types.ModuleType("tkinter.constants")
        for key, val in dict(RIDGE="ridge", BOTH="both", X="x", Y="y",
                             LEFT="left", RIGHT="right", TOP="top",
                             BOTTOM="bottom", N="n", S="s", E="e", W="w",
                             NW="nw", NE="ne", SW="sw", SE="se",
                             CENTER="center", YES=1, NO=0, END="end",
                             ALL="all").items():
            setattr(tkc, key, val)
        tk.constants = tkc
        itk = types.ModuleType("PIL.ImageTk")
        itk.PhotoImage = _Dummy
        sys.modules.setdefault("tkinter", tk)
        sys.modules.setdefault("tkinter.constants", tkc)
        sys.modules["PIL.ImageTk"] = itk
        import PIL
        PIL.ImageTk = itk
    from Watershed import Watershed

    class QuietWatershed(Watershed):
        def displayImage(self, argimage, title=""): pass
        def displayImage2(self, argimage, title=""): pass
        def displayImage3(self, argimage, title=""): pass
        def displayImage4(self, argimage, title=""): pass
        def displayImage5(self, argimage, title=""): pass
        def displayImage6(self, argimage, title=""): pass

    _QUIET = QuietWatershed
    return _QUIET


@contextlib.contextmanager
def _scratch():
    """The module reads its input from, and writes its intermediate pictures
    to, the current directory, and it insists on a bare file name.  Do all of
    that in a throw-away folder, and keep the module's chatter off screen."""
    here = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="hw7_")
    os.chdir(tmp)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield tmp
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)


def _binary_shed(mask):
    """A Watershed object whose data image is the boolean mask.  Call inside
    _scratch()."""
    on = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(on).save("mask.png")
    shed = _quiet_watershed()(data_image="mask.png",
                              binary_or_gray_or_color="binary")
    shed.extract_data_pixels()
    return shed


def module_erode(mask, r, shape="square"):
    """Erosion of a boolean mask by the module's erode(radius, shape).

    shape is "square" or "circular".  Returns a boolean array.  Read the
    module's erode() before you trust it: what it does at the edge of the
    picture is the subject of Task 2(a)."""
    with _scratch():
        shed = _binary_shed(mask)
        return np.array(shed.erode(shed.data_im, r, shape)).astype(bool)


def module_dilate(mask, r, shape="square"):
    """Dilation of a boolean mask by the module's dilate(radius, shape)."""
    with _scratch():
        shed = _binary_shed(mask)
        return np.array(shed.dilate(r, shape)).astype(bool)


def module_components(mask):
    """The module's connected_components(): 8-connected, two scans and a table
    of equivalences, which is the labeling of Rosenfeld and Kak, pages 241-242.

    The module's relabeling loop swaps rows and columns, so it only works on
    square pictures.  A non-square mask is padded with background to a square
    and cropped back.

    Returns:
        (labels, n): labels is 0 on the background and 1 ... n on the
        components
    """
    mask = np.asarray(mask) > 0
    h, w = mask.shape
    s = max(h, w)
    padded = np.zeros((s, s), bool)
    padded[:h, :w] = mask
    with _scratch():
        shed = _binary_shed(padded)
        shed.connected_components("data")
        labels = shed.componentID_array[:h, :w].copy()
        return labels, int(shed.number_of_components)


def count_holes(mask):
    """The number of holes in a boolean mask: the 4-connected components of
    the background that do not touch the border of the picture.  The
    foreground is 8-connected (as in module_components), so the background
    is 4-connected (Rosenfeld and Kak, page 207, ask for "opposite types of
    connectedness for S and its complement")."""
    lab, n = ndi.label(~np.asarray(mask, bool))    # default: 4-connected
    border = (set(lab[0, :]) | set(lab[-1, :])
              | set(lab[:, 0]) | set(lab[:, -1]))
    return len([k for k in range(1, n + 1) if k not in border])


def describe(mask):
    """(area in pixels, number of components, number of holes) of a mask."""
    mask = np.asarray(mask)
    return int(mask.sum()), module_components(mask)[1], count_holes(mask)


def influence_zones(mask, markers):
    """Influence zones of the markers inside the mask, grown by the module's
    own _unit_dilation_of_influence_zones(), one pixel at a time, until
    nothing changes.  This is what the module's influence_zones.py demo does
    with the marks you click.

    Args:
        mask: boolean array
        markers: int array, 0 = no marker, k = marker number k (k = 1, 2, ...)
    Returns:
        an int array: k on the influence zone of marker k, -1 on the geodesic
        skeleton (the lines where two zones meet), 0 outside the mask
    """
    pad = 2
    M = np.pad((np.asarray(mask) > 0).astype(int), pad)
    ##  Markers must lie in the mask: the ones outside are dropped here.
    K = np.pad(np.asarray(markers).astype(int), pad) * M
    labels = set(int(v) for v in np.unique(K) if v > 0)
    if not labels:
        return np.zeros(np.asarray(mask).shape, int)
    skiz = max(labels) + 1
    with _scratch():
        shed = _binary_shed(M)
        cur = K.copy()
        for _ in range(500):
            nxt = shed._unit_dilation_of_influence_zones(cur, M, labels)
            if nxt is None:
                break
            cur = nxt
    cur = cur[pad:-pad, pad:-pad]
    return np.where(cur == skiz, -1, cur)


def run_watershed(gray, sigma=1, thr=0.1, dec=16, marker_mask=None):
    """Prof. Kak's BLM watershed on a gray image (a 2-D uint8 array).  This is
    the sequence in his segment_automatic_and_show_watershed.py and, when
    marker_mask is given, in segment_with_markers_and_show_watershed.py.

    Args:
        sigma: Gaussian sigma applied before the gradient is taken
        thr: gradient_threshold_as_fraction; gradient values below thr * 255
            are set to zero
        dec: level_decimation_factor; only every dec-th gradient level is
            flooded
        marker_mask: optional boolean array.  Its True pixels become extra
            valleys (the gradient there is set to zero), which is what the
            module does with the polygons you would otherwise click.
    Returns:
        (labels, seconds): labels is an int array, the regions numbered 1, 2,
        ... and the watershed lines (the dams) -1
    """
    gray = np.asarray(gray, np.uint8)
    t0 = time.time()
    with _scratch():
        Image.fromarray(gray).save("g.png")
        shed = _quiet_watershed()(data_image="g.png",
                                  binary_or_gray_or_color="gray",
                                  size_for_calculations=max(gray.shape),
                                  sigma=sigma,
                                  gradient_threshold_as_fraction=thr,
                                  level_decimation_factor=dec, padding=0)
        shed.extract_data_pixels()
        shed.compute_gradient_image()
        if marker_mask is not None:
            valley = 1 - np.asarray(marker_mask).astype(int)
            shed.marked_valley_for_region = {0: valley}
            shed.modify_gradients_with_marker_minima()
        shed.compute_Z_level_sets_for_gradient_image()
        shed.propagate_influence_zones_from_bottom_to_top_of_Z_levels()
        labels = shed.componentID_array_dict[shed.max_grad_level - 1].copy()
    return labels, time.time() - t0


def evaluate(labels, truth, min_area=MIN_AREA):
    """Score a segmentation of the coins against the truth.

    Args:
        labels: int array.  Positive numbers (and 0, in an influence-zone map)
            are regions; negative numbers are dams.  A dam pixel belongs to no
            region: it is not counted for the table, for a coin or for the
            dust, and it is not part of any region's area.
        truth: coins_truth.png, 0 = table, 1 ... 15 = the coins

    The table is the largest region.  Every other region of at least
    min_area pixels is a candidate coin; smaller regions are dust and are
    left out altogether (of the count and of the scoring).  The Dice overlap
    of two sets A and B is 2|A and B| / (|A| + |B|), 1 for a perfect match
    and 0 for none.

    Returns:
        (n_regions, dice): n_regions is the number of candidate coins (the
        table, the dust and the dams are not counted).  dice is an array of
        15 numbers: dice[c - 1] is the best Dice overlap between the true coin
        c and any single candidate (0 if no candidate touches it).  Every coin
        looks for its own best candidate, so one region can be the best match
        of two coins: a merged pair is scored once for each coin.  The mean
        Dice of a segmentation is dice.mean(), over all 15 coins, zeros
        included.

    n_regions can be 15 for the wrong reasons (a merged pair of coins and a
    split coin cancel in the count), so look at dice as well; coins_found()
    below turns dice into a count that cannot be fooled that way.

    (describe() and module_components() count every connected component, dust
    included; evaluate() does not, so the same mask can have more components
    than candidate coins.)
    """
    labels = np.asarray(labels)
    ids = [int(v) for v in np.unique(labels) if v >= 0]
    area = {v: int((labels == v).sum()) for v in ids}
    table = max(area, key=area.get)
    regions = [v for v in ids if v != table and area[v] >= min_area]
    dice = []
    for c in range(1, int(truth.max()) + 1):
        A = truth == c
        best = 0.0
        for v in regions:
            B = labels == v
            inter = int((A & B).sum())
            if inter:
                best = max(best, 2.0 * inter / (A.sum() + B.sum()))
        dice.append(best)
    return len(regions), np.array(dice)


def coins_found(dice, thr=0.8):
    """How many of the 15 coins are FOUND: those whose best Dice overlap with a
    single region is at least thr.  A coin that is split in two, merged with
    its neighbor, or outlined badly is not found."""
    return int((np.asarray(dice) >= thr).sum())


def coin_of(pt, truth):
    """The coin (1 ... 15) under a point (row, col), 0 for the table.  Use it
    to ask whether your markers sit on the right coins.  The point is rounded
    to the nearest pixel with Python's round(), so a half goes to the even
    pixel: (62.0, 110.5) is read at (62, 110)."""
    return int(truth[int(round(pt[0])), int(round(pt[1]))])


def check_setup():
    """Check that the five data files are where DATA_DIR says, that Prof. Kak's
    Watershed module can be imported, and that its erode() gives the answer the
    rest of the homework assumes.  Run it before anything else.  It also
    prints the versions of Python, NumPy, SciPy, Pillow, Matplotlib and the
    module, which your README asks for.

    Prints one line per check and then check_setup: OK or check_setup: FAILED.

    Returns:
        True if all is well
    """
    import PIL
    import scipy
    ok = True
    for name in (COINS, TRUTH, RECT, DISCS, COINS_RGB):
        found = os.path.exists(os.path.join(DATA_DIR, name))
        if found:
            status = "found"
        else:
            status = "MISSING (looked in %s)" % os.path.abspath(DATA_DIR)
        print("  %-26s %s" % (name, status))
        ok = ok and found
    try:
        import matplotlib
        mpl = matplotlib.__version__
    except ImportError:
        mpl, ok = "MISSING (pip install matplotlib)", False
    versions = (("Python", sys.version.split()[0]), ("NumPy", np.__version__),
                ("SciPy", scipy.__version__), ("Pillow", PIL.__version__),
                ("Matplotlib", mpl))
    for name, version in versions:
        print("  %-26s %s" % (name, version))
    try:
        _quiet_watershed()
        ver = getattr(sys.modules["Watershed"], "__version__", "?")
        print("  %-26s version %s" % ("Watershed module", ver))
        block = np.zeros((9, 9), bool)
        block[2:7, 2:7] = True
        n = int(module_erode(block, 1, "square").sum())
        print("  %-26s a 5x5 block eroded once leaves %d pixels (expect 9)"
              % ("module erode()", n))
        ok = ok and n == 9
    except Exception as err:
        print("  %-26s NOT USABLE: %r (pip install Watershed)"
              % ("Watershed module", err))
        ok = False
    print("check_setup:", "OK" if ok else "FAILED")
    return ok


def colorize(labels, scale=3):
    """An RGB picture of a label image for the report: every region its own
    color, the table dark gray, the dams red.  Returns a PIL image."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(5)
    ids = [int(v) for v in np.unique(labels) if v >= 0]
    area = {v: int((labels == v).sum()) for v in ids}
    table = max(area, key=area.get)
    out = np.zeros(labels.shape + (3,), np.uint8)
    for v in ids:
        color = (30, 30, 30) if v == table else rng.integers(70, 255, 3)
        out[labels == v] = color
    out[labels == -1] = (255, 0, 0)
    size = (labels.shape[1] * scale, labels.shape[0] * scale)
    return Image.fromarray(out).resize(size, Image.NEAREST)


##  ============================================================================
##  Task 1: Otsu's threshold
##  ============================================================================

def otsu_threshold(gray):
    """Otsu's threshold of an 8-bit image.  The equations are (1)-(8) and
    (17)-(19) of Otsu (1979).

    Gray levels are 0 ... 255 here, where the paper uses 1 ... L.  Class C0 is
    every pixel with gray level <= k, class C1 every pixel above it.

    Returns:
        k_star: the k in 0 ... 254 that maximizes sigma_B^2(k)
        sigma_B2: float array of length 256.  sigma_B2[k] is sigma_B^2(k) of
            equation (18) wherever 0 < omega(k) < 1 (the paper's "effective
            range"), and 0 elsewhere
        p: the normalized histogram of equation (1), length 256
    
    Generated entirely by an AI assistant (Claude).
    """
    ##  TODO: from the histogram p, the cumulative omega(k) and mu(k) of
    ##  equations (6) and (7), the total mean mu_T of (8), and then
    ##  sigma_B^2(k) from (18) for every k at once.  Keep the k's with
    ##  omega(k) = 0 or 1 out of the argmax: there (18) is 0/0, numpy warns
    ##  about the division, and the NaN then wins np.argmax, which is silent.
    ##
    ##  No loop over the pixels, and no loop over k.  The histogram is one
    ##  call to np.bincount, and everything after it is a cumulative sum.
    #raise NotImplementedError
    gray = np.asarray(gray)
    counts = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = counts / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_T = mu[-1]
    valid = (omega > 0) & (omega < 1)
    sigma_B2 = np.zeros(256)
    sigma_B2[valid] = (mu_T * omega[valid] - mu[valid]) ** 2 / (
        omega[valid] * (1 - omega[valid]))
    k_star = int(np.argmax(np.where(valid, sigma_B2, -np.inf)))
    return k_star, sigma_B2, p

def class_report(gray, k):
    """Task 1(b): everything the paper defines for the threshold k.

    The class probabilities and means come from (2)-(5), the class variances
    from (10) and (11), and the within-class, between-class and total
    variances from (13), (14) and (15), each from its definition and NOT from
    equations (16) or (18).

    Returns:
        a dict with the keys omega0, omega1, mu0, mu1, var0, var1, sigma_W2,
        sigma_B2, sigma_T2 and eta (= sigma_B2 / sigma_T2, equation 17)
    """
    ##  TODO: see the docstring.  Nine lines.  Task 1(b) then asks you to
    ##  check equation (16) with the numbers you get here, so do not use (16)
    ##  to compute any of them: a check that uses what it checks proves
    ##  nothing.
    #raise NotImplementedError
    gray = np.asarray(gray)
    counts = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = counts / counts.sum()
    i = np.arange(256)
    mu_T = np.sum(i * p)
    omega0 = np.sum(p[:k + 1])
    omega1 = 1 - omega0
    mu0 = np.sum(i[:k + 1] * p[:k + 1]) / omega0
    mu1 = np.sum(i[k + 1:] * p[k + 1:]) / omega1
    var0 = np.sum((i[:k + 1] - mu0) ** 2 * p[:k + 1]) / omega0
    var1 = np.sum((i[k + 1:] - mu1) ** 2 * p[k + 1:]) / omega1
    sigma_W2 = omega0 * var0 + omega1 * var1
    sigma_B2 = omega0 * (mu0 - mu_T) ** 2 + omega1 * (mu1 - mu_T) ** 2
    sigma_T2 = np.sum((i - mu_T) ** 2 * p)
    eta = sigma_B2 / sigma_T2
    return dict(omega0=omega0, omega1=omega1, mu0=mu0, mu1=mu1,
                var0=var0, var1=var1, sigma_W2=sigma_W2, sigma_B2=sigma_B2,
                sigma_T2=sigma_T2, eta=eta)


def task1():
    """Task 1: Otsu's threshold, and what it does to the coins.

    Generated entirely by an AI assistant (Claude).
    """
    gray = load_gray()
    ##  TODO, following the parts of Task 1:
    ##    (a)  k* from otsu_threshold() and sigma_B^2(k*); the histogram (log
    ##         scale) beside the plot of sigma_B^2(k), with k* marked; from
    ##         the plot, whether the maximum is sharp or broad, and what that
    ##         means for choosing a threshold.
    ##    (b)  class_report() at k*; the check of equation (16) and its
    ##         residual; eta and what it says about this picture.
    ##    (c)  describe() of the mask gray > t for t = k*, 60, 140 and 180,
    ##         and what you see.
    #raise NotImplementedError
    k, sigma_B2, p = otsu_threshold(gray)
    print(f"k* = {k}")
    print(f"sigma_B2(k*) = {sigma_B2[k]:.4g}")

    r = class_report(gray, k)
    for key in ("omega0", "omega1", "mu0", "mu1", "var0", "var1",
                "sigma_W2", "sigma_B2", "sigma_T2", "eta"):
        print(f"{key:10s} = {r[key]:.4g}")
    residual = r["sigma_W2"] + r["sigma_B2"] - r["sigma_T2"]
    print("eq16 residual:", residual)

    import matplotlib.pyplot as plt
    counts = np.bincount(gray.ravel(), minlength=256)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(256), counts)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("gray level")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("histogram of coins.png")
    axes[1].plot(sigma_B2)
    axes[1].axvline(k, color="red", linestyle="--", label=f"k*={k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("sigma_B^2(k)")
    axes[1].set_title("between-class variance")
    axes[1].legend()
    save_figure(fig, "task1a_otsu.png")
    plt.close(fig)

    print(f"{'t':>4s} {'area':>6s} {'components':>11s} {'holes':>6s}")
    for t in (k, 60, 140, 180):
        mask = gray > t
        area, comps, holes = describe(mask)
        print(f"{t:4d} {area:6d} {comps:11d} {holes:6d}")


##  ============================================================================
##  Task 2: morphology
##  ============================================================================

def structuring_element_size(r, shape):
    """Task 2(a): the number of pixels in the module's structuring element of
    radius r, for shape "square" or "circular".

    Returns:
        an int
    """
    ##  TODO: a few lines.  Do not count the pixels from a formula, because
    ##  the formula is what Task 2(a) asks you to find out: dilate a single
    ##  pixel with module_dilate() and count what comes out.  Put the pixel
    ##  well inside the picture, so that nothing is cut off at its edge.
    #raise NotImplementedError
    picture = np.zeros((21, 21), bool)
    picture[10, 10] = True
    grown = module_dilate(picture, r, shape)
    return int(grown.sum())


def opening(mask, r, shape):
    """Task 2(b): erosion followed by dilation with the same element.

    Generated entirely by an AI assistant (Claude).
    """
    ##  TODO: one line, from module_erode() and module_dilate().
    #raise NotImplementedError
    return module_dilate(module_erode(mask, r, shape), r, shape)


def closing(mask, r, shape):
    """Task 2(c): dilation followed by erosion with the same element.

    Generated entirely by an AI assistant (Claude).
    """
    ##  TODO: one line.  Mind the order: it is the reverse of opening().
    #raise NotImplementedError
    return module_erode(module_dilate(mask, r, shape), r, shape)


def task2():
    """Task 2: erosion, dilation, opening and closing."""
    gray = load_gray()
    ##  TODO, following the parts of Task 2:
    ##    (a)  (i) structuring_element_size() for r = 1 ... 5 and both shapes.
    ##         (ii) erosion of an all-foreground 9 x 9 picture, with and
    ##         without a border of zeros added first; then the same erosion
    ##         with r = 3 of coin 14's mask, the Otsu mask restricted to the
    ##         pixels of coin 14 in the truth (this diagnostic is the one place
    ##         where the truth may be an input), and how many pixels survive
    ##         each time.
    ##    (b)  for r = 1, 3, 5, 7 and both shapes: area, number of components
    ##         and coins found of opening(mask, r, shape), mask = gray > k*
    ##         (coins found: module_components(), evaluate(), coins_found());
    ##         the best opening is the one that finds the most coins, a tie to
    ##         the smaller r; its picture (colorize()).
    ##    (c)  closing(load_binary(), r, shape) for both shapes: the smallest
    ##         r that repairs the outline (one component and one hole), found
    ##         by testing radii as needed, with (components, holes) at that r
    ##         and at the radius just below it; the two gaps, in pixels.

    import matplotlib.pyplot as plt
    truth = load_truth()
    k, _, _ = otsu_threshold(gray)
    mask = gray > k

    print("Task 2(a)(i): structuring element sizes")
    for shape in ("square", "circular"):
        sizes = [structuring_element_size(r, shape) for r in range(1, 6)]
        print(f"  {shape:9s} r=1..5: {sizes}")

    block = np.ones((9, 9), bool)
    plain = module_erode(block, 1, "square")
    padded = module_erode(np.pad(block, 1), 1, "square")[1:-1, 1:-1]
    print(f"Task 2(a)(ii): 9x9 eroded, no border {int(plain.sum())}, "
          f"with border {int(padded.sum())}")
    coin14 = mask & (truth == 14)     # the one allowed use of the truth
    plain = module_erode(coin14, 3, "square")
    padded = module_erode(np.pad(coin14, 3), 3, "square")[3:-3, 3:-3]
    print(f"  coin 14: {int(coin14.sum())} pixels, r=3 no border "
          f"{int(plain.sum())}, with border {int(padded.sum())}")

    print("Task 2(b): shape, r, area, components, found")
    best = None
    for shape in ("square", "circular"):
        for r in (1, 3, 5, 7):
            opened = opening(mask, r, shape)
            area, comps, _ = describe(opened)
            labels, _ = module_components(opened)
            _, dice = evaluate(labels, truth)
            found = coins_found(dice)
            print(f"  {shape:9s} {r:2d} {area:6d} {comps:4d} {found:4d}")
            if best is None or found > best[0] or (
                    found == best[0] and r < best[1]):
                best = (found, r, shape, labels, dice)
    found, r, shape, labels, dice = best
    print(f"  best opening: {shape} r={r}, found {found}, "
          f"Dice by coin {np.round(dice, 2)}")
    colorize(labels).save(os.path.join(OUT_DIR, "task2b_best_opening.png"))

    rect = load_binary()
    print(f"Task 2(c): broken rectangle {describe(rect)}")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rect, cmap="gray")
    axes[0].set_title("broken_rectangle1.jpg")
    for ax, shape in zip(axes[1:], ("square", "circular")):
        r = 1
        while describe(closing(rect, r, shape))[1:] != (1, 1):
            r += 1
        before = describe(closing(rect, r - 1, shape))
        after = describe(closing(rect, r, shape))
        print(f"  {shape}: r={r - 1} (comps, holes) = {before[1:]}, "
              f"r={r} (comps, holes) = {after[1:]}")
        ax.imshow(closing(rect, r, shape), cmap="gray")
        ax.set_title(f"closing, {shape}, r={r}")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task2c_closing.png")
    plt.close(fig)

    rows = np.flatnonzero(rect.any(axis=1))
    cols = np.flatnonzero(rect.any(axis=0))
    top = rect[rows[0]:rows[0] + 2, cols[0]:cols[-1] + 1].any(axis=0)
    right = rect[rows[0]:rows[-1] + 1, cols[-1] - 1:cols[-1] + 1].any(axis=1)
    print(f"  gap in the top side: {int((~top).sum())} pixels, "
          f"gap in the right side: {int((~right).sum())} pixels")


##  ============================================================================
##  Task 3: distance maps, and one marker per coin
##  ============================================================================

def distance_two_pass(mask, connectivity, passes=2):
    """The distance of every foreground pixel from the nearest background
    pixel, by the two-scan algorithm of Rosenfeld and Kak, page 214, equation
    (2).  Task 3(a) writes it and checks it on the diamond; Task 3(b) runs it
    on the Otsu mask.

    Args:
        mask: boolean array, True = foreground
        connectivity: 4 for the city-block distance d4, 8 for the chessboard
            distance d8
        passes: 2 for the finished distance map, 1 for the array after the
            forward scan only, which is what Task 3(a) asks you to show
    Returns:
        an int array, 0 on the background
    """
    ##  TODO: the book assumes that the picture is surrounded by background,
    ##  so add a border of zeros one pixel wide before you start and take it
    ##  off at the end.  Start every foreground pixel at a number larger than
    ##  any distance can be, and every background pixel at 0.  In the first
    ##  scan a pixel takes min(itself, 1 + the smallest value among its
    ##  neighbors that PRECEDE it in the scan); in the second scan, the same
    ##  with the neighbors that FOLLOW it.  For connectivity 4 the neighbors
    ##  that precede a pixel are the one above and the one to the left; for 8
    ##  add the two upper diagonals.
    ##
    ##  The two scans are loops over the pixels, and that is fine: it takes a
    ##  few hundredths of a second.  Nothing else in this homework needs a
    ##  loop over the pixels.
    ##
    big = mask.size                          
    D = np.where(np.pad(np.asarray(mask, bool), 1), big, 0)
    before = [(-1, 0), (0, -1)]               
    if connectivity == 8:
        before += [(-1, -1), (-1, 1)]         
    after = [(-dr, -dc) for dr, dc in before] 
    rows, cols = D.shape

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if D[r, c]:
                D[r, c] = min(D[r, c],
                              1 + min(D[r + dr, c + dc] for dr, dc in before))
    if passes == 2:
        for r in range(rows - 2, 0, -1):
            for c in range(cols - 2, 0, -1):
                if D[r, c]:
                    D[r, c] = min(D[r, c],
                                  1 + min(D[r + dr, c + dc] for dr, dc in after))
    return D[1:-1, 1:-1]


def local_maxima(D, w, dmin=1):
    """Task 3(c): a boolean array that is True where D equals its maximum over
    the (2w+1) x (2w+1) window around the pixel, and D >= dmin.  w = 1 is the
    comparison with the 8 neighbors.  Rosenfeld and Kak (page 211) compare d4
    with the 4 neighbors and d8 with the 8; one square window serves both
    here, and widening it is what Task 3(c) is about.
    """
    ##  TODO: two lines.  scipy.ndimage.maximum_filter is allowed here.  The
    ##  book assumes the picture is surrounded by background (page 214), so
    ##  mode="constant", cval=0 is the natural choice; on this picture every
    ##  edge mode gives the same markers.
    #raise NotImplementedError
    biggest = ndi.maximum_filter(D, size=2 * w + 1, mode="constant", cval=0)
    return (D == biggest) & (D >= dmin)

def find_markers(D, w=5, dmin=3):
    """Task 3(c): one marker for every group of touching local maxima.

    Labels the 8-connected groups of local_maxima(D, w, dmin) with
    module_components() (8-connected for D4 as well as for D8) and describes
    each group by its center.  The labels are what Tasks 4(a) and 4(d) use;
    the centers are for reporting, for coin_of() and for the discs of Task
    4(c).

    Returns:
        (marker_labels, centers): marker_labels is an int array, 0 or k = 1,
        2, ...; EVERY pixel of group k is labeled k, so a marker is the whole
        group and not one pixel of it.  centers is a list of (row, col, d)
        with the center of group k (the mean of its pixel coordinates, NOT
        rounded) and d, the largest value of D in the group
    """
    ##  TODO: about six lines.
    marker_labels, n = module_components(local_maxima(D, w, dmin))
    centers = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(marker_labels == k)
        centers.append((rr.mean(), cc.mean(), int(D[rr, cc].max())))
    return marker_labels, centers


def task3():
    """Task 3: distance maps and markers."""
    gray = load_gray()
    truth = load_truth()
    ##  TODO, following the parts of Task 3:
    ##    (a)  distance_two_pass(DIAMOND, 4, passes=1), distance_two_pass(
    ##         DIAMOND, 4) and distance_two_pass(DIAMOND, 8), checked against
    ##         what you worked out on paper; then the local maxima of the
    ##         final d4 array (no larger 4-neighbor) and of the final d8 array
    ##         (no larger 8-neighbor), and how many there are in each.
    ##    (b)  D4 and D8 of the Otsu mask: the maximum and the sum of each;
    ##         the check against scipy (give it the same padded mask).
    ##    (c)  local_maxima() and find_markers() on D4 for w = 1, 3, 5, 7
    ##         with dmin = 3, then the same on D8: the number of local-maximum
    ##         pixels, the number of markers, the coins with more than one
    ##         marker (coin_of()) and the markers on the table; the positions
    ##         of the extra D8 markers; the (row, col, d) of the 15 markers you
    ##         keep.
    np.set_printoptions(linewidth=120)
    d4_first = distance_two_pass(DIAMOND, 4, passes=1)
    d4 = distance_two_pass(DIAMOND, 4)
    d8 = distance_two_pass(DIAMOND, 8)
    print("Task 3(a): d4 after the forward scan\n", d4_first)
    print("final d4\n", d4)
    print("final d8\n", d8)

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    for name, D, foot in (("d4", d4, cross), ("d8", d8, np.ones((3, 3), bool))):
        peak = (D == ndi.maximum_filter(D, footprint=foot, mode="constant")) \
            & (D > 0)
        print(f"  local maxima of {name}: {int(peak.sum())}\n",
              peak.astype(int))

    k, _, _ = otsu_threshold(gray)
    mask = gray > k
    padded = np.pad(mask, 1)
    maps = {}
    for conn, metric in ((4, "taxicab"), (8, "chessboard")):
        D = distance_two_pass(mask, conn)
        ref = ndi.distance_transform_cdt(padded, metric=metric)[1:-1, 1:-1]
        maps[conn] = D
        print(f"Task 3(b): D{conn} max {int(D.max())}, sum {int(D.sum())}, "
              f"agrees with scipy {metric}: {np.array_equal(D, ref)}")


    print("Task 3(c): map, w, maxima pixels, markers, coins with >1 marker, "
          "markers on the table")
    for conn in (4, 8):
        for w in (1, 3, 5, 7):
            peaks = local_maxima(maps[conn], w, 3)
            labels, centers = find_markers(maps[conn], w, 3)
            coins = [coin_of(c, truth) for c in centers]
            multi = sorted(c for c in set(coins) if c and coins.count(c) > 1)
            table = coins.count(0)
            print(f"  D{conn} w={w}: {int(peaks.sum()):4d} {len(centers):4d} "
                  f"{multi} {table}")
            if conn == 8 and w > 1:
                extra = [(c, coin) for c, coin in zip(centers, coins)
                         if coin in multi]
                for c, coin in extra:
                    print(f"      on coin {coin:2d}: ({c[0]:.1f}, {c[1]:.1f}),"
                          f" d={c[2]}")


    labels, centers = find_markers(maps[4], 5, 3)
    print("chosen setting D4, w=5, dmin=3: (row, col, d) and coin")
    for c in centers:
        print(f"  ({c[0]:.2f}, {c[1]:.2f}, {c[2]:2d})  coin {coin_of(c, truth)}")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, conn in zip(axes[:2], (4, 8)):
        im = ax.imshow(maps[conn], cmap="viridis")
        ax.set_title(f"D{conn} of the Otsu mask")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[2].imshow(mask, cmap="gray")
    axes[2].contour(labels > 0, levels=[0.5], colors="red")
    for c in centers:
        axes[2].plot(c[1], c[0], "r+", markersize=8)
    axes[2].set_title("D4 markers, w=5, dmin=3")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task3_distance_markers.png")
    plt.close(fig)


##  ============================================================================
##  Task 4: the watershed
##  ============================================================================

def marker_discs(shape, centers, rho):
    """Task 4(c): the markers as discs, not as the few pixels at the maximum.
    A boolean array of the given shape that is True where
    (row - cy)**2 + (col - cx)**2 <= (rho * d)**2 for at least one marker
    (cy, cx, d) of centers.  d is the distance-map value at the marker, which
    is about the coin's radius, so rho = 0.5 is a disc of half the coin.
    """
    ##  TODO: four lines.
    #raise NotImplementedError
    rr, cc = np.indices(shape)
    out = np.zeros(shape, bool)
    for cy, cx, d in centers:
        out |= (rr - cy) ** 2 + (cc - cx) ** 2 <= (rho * d) ** 2
    return out


def split_merged_regions(ws_labels, marker_labels):
    """Task 4(d): the hybrid.  Wherever one region of ws_labels holds two or
    more markers, replace that region by the influence zones of its markers
    (influence_zones() above, with the region as the mask).  Regions with one
    marker or none are left alone.

    Args:
        ws_labels: the label image that run_watershed() returned (the first
            item of its result): regions 1, 2, ..., dams -1
        marker_labels: the label image that find_markers() returned: 0, 1,
            ..., m (the markers of Task 3(c), NOT the discs of Task 4(c))

    The rules, so that everyone gets the same picture:
      * A marker belongs to the one region that holds most of its pixels.  A
        dam pixel (-1) is in no region, so it does not count, and a tie goes
        to the smaller label.  A marker with no pixel in any region belongs
        to none.
      * Only the regions that hold two or more markers change.  Every other
        pixel keeps its label, and that includes every dam pixel outside the
        regions that were split.
      * Inside a region that is split, the influence zones of its markers
        replace it: each zone gets a new number above the old ones, and the
        lines between the zones are -1.

    Returns:
        the new label image
    """
    ##  TODO: find the region of every marker (np.bincount of the region
    ##  labels under its pixels, dams left out, is one way), then for every
    ##  region that holds two or more markers call influence_zones() on that
    ##  region alone, with only those markers.  About 15 lines.
    #raise NotImplementedError

    out = ws_labels.copy()
    # which watershed region does each marker belong to?
    owner = {}
    for k in range(1, int(marker_labels.max()) + 1):
        under = ws_labels[marker_labels == k]
        under = under[under > 0]                    # dam pixels do not count
        if under.size:
            owner[k] = int(np.bincount(under).argmax())   # ties -> smaller label
    offset = int(ws_labels.max())                   # new labels start above this
    for reg in set(owner.values()):
        ks = [k for k, r in owner.items() if r == reg]
        if len(ks) < 2:
            continue                                # one marker: leave alone
        region = ws_labels == reg
        mk = np.where(np.isin(marker_labels, ks), marker_labels, 0)
        zones = influence_zones(region, mk)
        out[region & (zones > 0)] = zones[region & (zones > 0)] + offset
        out[region & (zones == -1)] = -1
    return out

def task4():
    """Task 4: influence zones, the automatic watershed, marker control and
    the hybrid."""
    gray = load_gray()
    truth = load_truth()
    ##  TODO, following the parts of Task 4:
    ##    (a)  influence_zones(mask, markers) with the 15 markers of Task 3;
    ##         regions, coins_found(), mean Dice, the number of dam pixels and
    ##         the Dice of every coin; show it with colorize(), and look at
    ##         which coins score worst.
    ##    (b)  run_watershed(gray, sigma, thr, 16) for sigma in {1, 2} and
    ##         thr in {0.1, 0.3}; regions, coins_found() and mean Dice for
    ##         each; the picture for the default (1, 0.1) and for the best of
    ##         the four.
    ##    (c)  run_watershed(..., marker_mask=marker_discs(...)) for rho in
    ##         {0.3, 0.4, 0.6, 0.9}; regions, coins_found(), mean Dice and
    ##         the worst coin; the picture and the Dice by coin for the best
    ##         rho, and what happens to coins 8 and 9.
    ##    (d)  split_merged_regions() on the result of (c): its regions, coins
    ##         found, mean Dice and picture; the scoreboard, with the Otsu
    ##         mask and its best opening from Task 2 added.
    ##
    ##  Where a part asks for the best run, take the highest mean Dice; a tie
    ##  goes to the run listed first in your table.  Each run_watershed() takes
    ##  about five seconds, so save the label images with np.save while you
    ##  develop.  Your submitted code must still make them from scratch.
    #raise NotImplementedError
    gray = load_gray()
    truth = load_truth()
    k, _, _ = otsu_threshold(gray)
    mask = gray > k
    mk, centers = find_markers(distance_two_pass(mask, 4), 5, 3)
    rows = []                                       # the scoreboard

    def score(name, labels):
        n, dice = evaluate(labels, truth)
        rows.append((name, n, coins_found(dice), dice.mean()))
        print(f"{name}: regions {n}, found {coins_found(dice)}, "
              f"mean Dice {dice.mean():.3f}")
        return dice

    def cached(name, **kw):                         # a run takes ~5 s
        path = f"output/{name}.npy"
        if os.path.exists(path):
            return np.load(path)
        labels, _ = run_watershed(gray, dec=16, **kw)
        np.save(path, labels)
        return labels

    # rows 1 and 2 of the scoreboard come from Task 2
    score("Otsu mask, components", module_components(mask)[0])
    score("opening, best of 2(b)", module_components(opening(mask, 5, "square"))[0])

    # (a) influence zones of the markers
    iz = influence_zones(mask, mk)
    dice = score("influence zones", iz)
    print("  dam pixels:", int((iz == -1).sum()))
    print("  Dice by coin:", np.round(dice, 2),
          "worst two coins:", np.argsort(dice)[:2] + 1)
    colorize(iz).save("output/task4a_iz.png")

    # (b) automatic watershed
    best = None
    for sigma in (1, 2):
        for thr in (0.1, 0.3):
            lab = cached(f"auto_{sigma}_{thr}", sigma=sigma, thr=thr)
            d = score(f"auto sigma={sigma} thr={thr}", lab)
            if best is None or d.mean() > best[0]:   # strict > : tie keeps first
                best = (d.mean(), lab, sigma, thr)
            if (sigma, thr) == (1, 0.1):
                default = lab
    rows[-4:] = [rows[-4], rows[-4]] and rows[-4:]   # keep the 4 rows as run
    colorize(default).save("output/task4b_default.png")
    colorize(best[1]).save("output/task4b_best.png")
    auto_default = next(r for r in rows if r[0] == "auto sigma=1 thr=0.1")
    auto_best = max(rows[-4:], key=lambda r: r[3])

    # (c) marker discs
    top = None
    for rho in (0.3, 0.4, 0.6, 0.9):
        lab = cached(f"marker_{rho}", sigma=1, thr=0.1,
                     marker_mask=marker_discs(gray.shape, centers, rho))
        d = score(f"marker rho={rho}", lab)
        print("  worst coin:", int(np.argmin(d)) + 1, f"({d.min():.2f})")
        if top is None or d.mean() > top[0]:
            top = (d.mean(), lab, rho, d)
    print("best rho", top[2], "Dice by coin", np.round(top[3], 2))
    colorize(top[1]).save("output/task4c_best.png")
    marker_best = next(r for r in rows if r[0] == f"marker rho={top[2]}")

    # (d) hybrid
    hyb = split_merged_regions(top[1], mk)
    score("marker watershed + influence zones", hyb)
    colorize(hyb).save("output/task4d_hybrid.png")

    print("\nScoreboard")
    final = [rows[0], rows[1], next(r for r in rows if r[0] == "influence zones"),
             auto_default, auto_best, marker_best, rows[-1]]
    for name, n, f, m in final:
        print(f"  {name:38s} {n:3d} {f:3d} {m:.3f}")

##  ============================================================================
##  Task 5: color, and a pretrained segmenter
##  ============================================================================

def task5():
    """Task 5: Otsu's threshold on the channels of discs_rgb.png.  Part (b) is
    done in a browser, on coins_rgb.png: there is no code for it."""
    rgb = load_rgb()
    ##  TODO, for part (a) of Task 5:
    ##    Make four pictures: the gray picture, rgb_to_gray(rgb), and the
    ##    channels rgb[..., 0], rgb[..., 1] and rgb[..., 2].  Apply your
    ##    otsu_threshold() to the gray picture and to each channel and make
    ##    the mask level > k* for each; if a mask holds the table and not the
    ##    discs, invert it and say so.  Report k*, and the area and the number
    ##    of connected components of the mask you keep, after any inversion
    ##    (describe() gives both).  Print the mean level of an 11 x 11 patch
    ##    centered at (row, col) = (10, 10) on the table, (64, 42) on the red
    ##    disc and (64, 86) on the green one, in gray, R, G and B.  Show the
    ##    input, the gray picture and the four masks in one figure
    ##    (save_figure()).
    #raise NotImplementedError
    rgb = load_rgb()
    pics = {"gray": rgb_to_gray(rgb), "R": rgb[..., 0],
            "G": rgb[..., 1], "B": rgb[..., 2]}
    spots = {"table": (10, 10), "red disc": (64, 42), "green disc": (64, 86)}

    masks = {}
    for name, pic in pics.items():
        k, _, _ = otsu_threshold(pic)
        mask = pic > k
        # if the table patch is mostly in the mask, the mask holds the table

def task5():
    rgb = load_rgb()
    pics = {"gray": rgb_to_gray(rgb), "R": rgb[..., 0],
            "G": rgb[..., 1], "B": rgb[..., 2]}
    spots = {"table": (10, 10), "red disc": (64, 42), "green disc": (64, 86)}

    masks = {}
    for name, pic in pics.items():
        k, _, _ = otsu_threshold(pic)
        mask = pic > k
        # if the table patch is mostly in the mask, the mask holds the table
        inverted = bool(mask[5:16, 5:16].mean() > 0.5)
        if inverted:
            mask = ~mask
        masks[name] = mask
        area, comps, holes = describe(mask)
        print(f"{name:5s}: k*={k}, area {area}, components {comps}, "
              f"inverted: {inverted}")

    print("mean of the 11x11 patch (gray, R, G, B)")
    for where, (r, c) in spots.items():
        means = [pic[r - 5:r + 6, c - 5:c + 6].mean() for pic in pics.values()]
        print(f"  {where:10s}", " ".join(f"{m:6.1f}" for m in means))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("input")
    axes[0, 1].imshow(pics["gray"], cmap="gray")
    axes[0, 1].set_title("gray")
    for ax, name in zip((axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]),
                        ("gray", "R", "G", "B")):
        ax.imshow(masks[name], cmap="gray")
        ax.set_title(f"mask of {name}")
    for ax in axes.ravel():
        ax.axis("off")
    save_figure(fig, "task5a_channels.png")
    plt.close(fig)

##  ============================================================================
##  Driver
##  ============================================================================

def main():
    if not check_setup():
        raise SystemExit("Fix the setup first.")
    task1()
    task2()
    task3()
    task4()
    task5()


if __name__ == '__main__':
    main()

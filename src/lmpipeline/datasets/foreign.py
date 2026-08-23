"""Recognize an upload that belongs to a DIFFERENT DIMER pipeline, so the user is told so.

A vision dataset uploaded to the language-model pipeline currently fails with "required
split 'train' not found", which is true and useless: it describes what this pipeline wanted
rather than what the user actually did. DIMER's Common Errors guidance names "wrong pipeline
selected" as a top user error and asks the validator to recognize the common mismatches and
say which pipeline the upload looks like it belongs to.

Three properties are load-bearing, and all three are about NOT overstepping:

  * STRUCTURE ONLY. Detection reads directory names, file extensions and file counts. It
    never opens a file. That keeps it cheap on a large upload, keeps a malformed or hostile
    file from being parsed on a path that exists to produce a friendly message, and keeps
    user content out of the result document by construction rather than by remembering to
    redact it.
  * ABSTAIN WHEN AMBIGUOUS. A wrong guess is worse than the generic error, because it sends
    the user to another pipeline that will also reject them. Every rule below requires an
    unmistakable signature, and anything short of one returns None so the ordinary
    split-missing error stands.
  * ADVISORY, NEVER AUTHORITATIVE. Nothing here decides whether a dataset is valid. It only
    replaces one rejection message with a more useful rejection message.

The signatures are the conventional on-disk layouts of the vision tasks DIMER documents:
imagefolder classification, image/mask segmentation pairs, and the COCO/VOC/YOLO detection
layouts. They are deliberately the layouts as PUBLISHED BY THE TOOLS, not guesses about what
a user might do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Extensions that make a file an image for our purposes. Deliberately the common ones: an
# exhaustive list would add exotic formats that, if seen, are more likely to mean something
# unusual is going on than that this is a vision dataset.
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
)

# Annotation-carrying extensions, by convention: COCO ships .json, VOC ships .xml, YOLO
# ships .txt. Recognized by extension only -- none of them is opened.
ANNOTATION_SUFFIXES = frozenset({".json", ".xml", ".txt", ".csv"})

# Directory names the vision ecosystems settled on. Lowercased before comparison.
IMAGE_DIR_NAMES = frozenset({"images", "image", "img", "imgs", "jpegimages", "pics"})
MASK_DIR_NAMES = frozenset(
    {"masks", "mask", "labels_masks", "segmentation", "segmentations",
     "segmentationclass", "gt", "ground_truth", "annotations_masks"}
)
ANNOTATION_DIR_NAMES = frozenset({"annotations", "annotation", "labels", "label"})

_IGNORED_DIRS = frozenset({"__MACOSX", ".git", ".ipynb_checkpoints", "__pycache__"})

# How much evidence is enough. A couple of stray images -- a logo, a diagram in a README --
# must not reclassify a language-model upload, so the threshold sits above incidental use.
MIN_IMAGES = 4
MIN_CLASS_DIRS = 2

# Traversal bound. Detection runs on an upload that has ALREADY failed validation, so it must
# stay cheap: a wrong-pipeline upload can be a directory of a hundred thousand images. The
# thresholds above are met long before this cap, so the cap costs no accuracy.
MAX_ENTRIES_SCANNED = 20_000

# Task names as this pipeline reports them. Descriptive rather than an assertion about
# DIMER's own pipeline identifiers, which are a portal-side fact this repository has not
# read out and must not invent -- COMPATIBILITY.md's rule about unmeasured constants.
TASK_IMAGE_CLASSIFICATION = "image classification"
TASK_IMAGE_SEGMENTATION = "image segmentation"
TASK_OBJECT_DETECTION = "object detection"


@dataclass(frozen=True)
class ForeignDataset:
    """A confident structural identification of an upload from another pipeline."""

    task: str
    reason: str
    evidence: dict = field(default_factory=dict)

    def message(self) -> str:
        return (
            f"This upload looks like an {self.task} dataset, not a language-model "
            f"fine-tuning dataset: {self.reason}. It was most likely submitted to the wrong "
            "DIMER pipeline -- select the pipeline for that task instead. If it really is a "
            "language-model dataset, it must contain train.jsonl at the top level, with one "
            "JSON object per line."
        )


@dataclass
class _Survey:
    """What a bounded structural walk of the upload found. Names and counts only."""

    image_files: int = 0
    jsonl_files: int = 0
    other_files: int = 0
    suffixes: Counter = field(default_factory=Counter)
    # Top-level directory name -> what it directly contains.
    dir_images: Counter = field(default_factory=Counter)
    dir_annotations: Counter = field(default_factory=Counter)
    dir_stems: dict = field(default_factory=dict)
    truncated: bool = False


def _survey(root: Path) -> _Survey:
    """Walk the tree once, bounded, recording structure and nothing else."""
    survey = _Survey()
    scanned = 0

    for path in root.rglob("*"):
        if scanned >= MAX_ENTRIES_SCANNED:
            survey.truncated = True
            break
        try:
            relative = path.relative_to(root)
        except ValueError:  # pragma: no cover - rglob yields only descendants
            continue
        if any(part in _IGNORED_DIRS for part in relative.parts):
            continue
        scanned += 1
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        survey.suffixes[suffix] += 1
        if suffix == ".jsonl":
            survey.jsonl_files += 1
        elif suffix in IMAGE_SUFFIXES:
            survey.image_files += 1
        else:
            survey.other_files += 1

        # Attribute the file to the top-level directory it sits under, which is the level
        # every one of these layouts organizes itself at.
        if len(relative.parts) < 2:
            continue
        top = relative.parts[0]
        if suffix in IMAGE_SUFFIXES:
            survey.dir_images[top] += 1
            survey.dir_stems.setdefault(top, set()).add(path.stem)
        elif suffix in ANNOTATION_SUFFIXES:
            survey.dir_annotations[top] += 1
            survey.dir_stems.setdefault(top, set()).add(path.stem)

    return survey


def _named(candidates: Counter, names: frozenset) -> list[str]:
    return sorted(d for d in candidates if d.lower() in names)


def _detect_segmentation(survey: _Survey) -> ForeignDataset | None:
    """An images/ + masks/ pair whose members correspond by filename.

    The correspondence is what makes this segmentation rather than two unrelated folders,
    and it is checked on stems -- filenames, never contents.
    """
    image_dirs = _named(survey.dir_images, IMAGE_DIR_NAMES)
    mask_dirs = _named(survey.dir_images, MASK_DIR_NAMES)
    if not image_dirs or not mask_dirs:
        return None

    image_dir, mask_dir = image_dirs[0], mask_dirs[0]
    shared = survey.dir_stems.get(image_dir, set()) & survey.dir_stems.get(mask_dir, set())
    if len(shared) < MIN_IMAGES:
        return None

    return ForeignDataset(
        task=TASK_IMAGE_SEGMENTATION,
        reason=(
            f"it pairs a {image_dir}/ directory with a {mask_dir}/ directory sharing "
            f"{len(shared)} filenames, which is the conventional segmentation layout"
        ),
        evidence={
            "imageDirectory": image_dir,
            "maskDirectory": mask_dir,
            "pairedFilenames": len(shared),
            "imageFiles": survey.image_files,
        },
    )


def _detect_object_detection(survey: _Survey) -> ForeignDataset | None:
    """Images alongside a conventional annotation directory (COCO, VOC or YOLO)."""
    image_dirs = _named(survey.dir_images, IMAGE_DIR_NAMES)
    annotation_dirs = _named(survey.dir_annotations, ANNOTATION_DIR_NAMES)
    if not image_dirs or not annotation_dirs:
        return None
    if survey.image_files < MIN_IMAGES:
        return None

    annotation_dir = annotation_dirs[0]
    # The annotation directory must actually be annotations rather than more images -- an
    # `annotations/` full of PNGs is segmentation, and _detect_segmentation ran first.
    if survey.dir_images.get(annotation_dir, 0) > survey.dir_annotations[annotation_dir]:
        return None

    kinds = sorted(
        suffix for suffix in (".json", ".xml", ".txt", ".csv") if survey.suffixes.get(suffix)
    )
    return ForeignDataset(
        task=TASK_OBJECT_DETECTION,
        reason=(
            f"it holds {survey.image_files} image files beside a {annotation_dir}/ "
            f"directory of {', '.join(kinds) or 'annotation'} files, which is the "
            "conventional detection layout"
        ),
        evidence={
            "imageDirectory": image_dirs[0],
            "annotationDirectory": annotation_dir,
            "annotationFiles": survey.dir_annotations[annotation_dir],
            "imageFiles": survey.image_files,
        },
    )


def _detect_image_classification(root: Path, survey: _Survey) -> ForeignDataset | None:
    """The imagefolder convention: one directory per class, images inside each.

    Runs last because an images/masks pair also satisfies "two directories of images"; the
    layouts above are more specific and claim their uploads first.
    """
    if survey.image_files < MIN_IMAGES:
        return None

    class_dirs = sorted(
        name for name, count in survey.dir_images.items()
        if count >= 1 and survey.dir_annotations.get(name, 0) == 0
    )
    if len(class_dirs) < MIN_CLASS_DIRS:
        return None

    # A split-style wrapper (train/ val/ test/ each holding class directories) is the same
    # convention one level down, and reads as classification just as unambiguously.
    top_level_files = [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES
        and p.name not in ("README", "README.md", "LICENSE", "classes.txt")
    ]
    if top_level_files:
        return None

    return ForeignDataset(
        task=TASK_IMAGE_CLASSIFICATION,
        reason=(
            f"it holds {survey.image_files} image files spread across {len(class_dirs)} "
            "directories with no annotation files, which is the imagefolder convention for "
            "per-class image directories"
        ),
        evidence={
            # Bounded, and directory names only -- these are structure, not payload.
            "directories": class_dirs[:5],
            "directoryCount": len(class_dirs),
            "imageFiles": survey.image_files,
        },
    )


def detect_foreign_dataset(root: Path) -> ForeignDataset | None:
    """Identify an upload belonging to another DIMER pipeline, or return None.

    None means "no confident identification" and is the correct answer whenever the
    structure is ambiguous. The caller keeps its ordinary error in that case.
    """
    root = Path(root)
    if not root.is_dir():
        return None

    survey = _survey(root)

    # A .jsonl anywhere means this is a language-model dataset with a layout or naming
    # problem, which the ordinary split-resolution errors already diagnose precisely. Saying
    # "this looks like a vision dataset" over the top of that would be actively misleading.
    if survey.jsonl_files:
        return None

    # The walk above stops at MAX_ENTRIES_SCANNED, so on a very large upload "no .jsonl
    # seen" is not the same as "no .jsonl present" -- and the difference decides whether
    # this abstains. Confirm it, lazily: rglob yields as it walks and next() stops at the
    # first hit, so the cost lands on the case that turns out to BE a language-model
    # dataset. A second walk here is cheap next to telling a user their dataset belongs to
    # a pipeline it does not.
    if survey.truncated and next(root.rglob("*.jsonl"), None) is not None:
        return None

    # Most specific layout first: segmentation and detection both look like classification
    # if you only count directories of images.
    for detector in (_detect_segmentation, _detect_object_detection):
        found = detector(survey)
        if found is not None:
            return found

    return _detect_image_classification(root, survey)

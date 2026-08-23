"""Wrong-pipeline detection must be confident when it speaks and silent when it is not.

Issue (validator #5) comes from DIMER's Common Errors guidance: "wrong pipeline selected" is
a top user error, and the validator should recognize the common mismatches and name the
likely task rather than reporting a missing train.jsonl.

The risk this carries is a WRONG suggestion, which is worse than the generic error because
it sends the user to a second pipeline that will also reject them. So the abstention cases
below are not filler -- they are the half of the feature that keeps it honest, and there are
deliberately more of them than positive cases.
"""

from __future__ import annotations

import pytest

from lmpipeline.datasets.foreign import (
    TASK_IMAGE_CLASSIFICATION,
    TASK_IMAGE_SEGMENTATION,
    TASK_OBJECT_DETECTION,
    detect_foreign_dataset,
)
from lmpipeline.datasets.resolver import resolve_dataset
from lmpipeline.errors import Code, DatasetError


def _images(directory, count, suffix=".jpg", prefix="img"):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"{prefix}{i}{suffix}").write_bytes(b"\x89PNG-not-really")


def _imagefolder(root):
    """The torchvision/HF `imagefolder` convention: one directory per class."""
    _images(root / "cats", 6)
    _images(root / "dogs", 5)
    return root


def _segmentation(root):
    _images(root / "images", 6, prefix="scene")
    _images(root / "masks", 6, suffix=".png", prefix="scene")
    return root


def _detection(root):
    _images(root / "images", 8, prefix="frame")
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for i in range(8):
        (root / "annotations" / f"frame{i}.xml").write_text("<annotation/>", encoding="utf-8")
    return root


# -- confident identifications -------------------------------------------------------


def test_an_imagefolder_upload_is_recognized_as_classification(tmp_path):
    found = detect_foreign_dataset(_imagefolder(tmp_path))
    assert found is not None
    assert found.task == TASK_IMAGE_CLASSIFICATION
    assert found.evidence["directoryCount"] == 2
    assert found.evidence["imageFiles"] == 11


def test_an_image_mask_pair_is_recognized_as_segmentation(tmp_path):
    found = detect_foreign_dataset(_segmentation(tmp_path))
    assert found is not None
    assert found.task == TASK_IMAGE_SEGMENTATION
    assert found.evidence["pairedFilenames"] == 6


def test_images_beside_annotations_are_recognized_as_detection(tmp_path):
    found = detect_foreign_dataset(_detection(tmp_path))
    assert found is not None
    assert found.task == TASK_OBJECT_DETECTION


def test_segmentation_is_not_mistaken_for_classification(tmp_path):
    """Two directories of images is also the imagefolder signature; specificity must win."""
    found = detect_foreign_dataset(_segmentation(tmp_path))
    assert found.task == TASK_IMAGE_SEGMENTATION


def test_the_message_names_the_task_and_still_says_what_this_pipeline_needs(tmp_path):
    message = detect_foreign_dataset(_imagefolder(tmp_path)).message()
    assert TASK_IMAGE_CLASSIFICATION in message
    assert "train.jsonl" in message
    assert "wrong DIMER pipeline" in message


# -- abstentions ---------------------------------------------------------------------


def test_a_malformed_language_model_dataset_is_never_reclassified(tmp_path):
    """The regression that would make this feature a net loss.

    A user whose train.jsonl is nested one level too deep, or misnamed, must keep the error
    that tells them exactly that. A .jsonl anywhere is enough to make this an LM dataset with
    a layout problem.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    assert detect_foreign_dataset(tmp_path) is None


def test_an_lm_dataset_with_a_few_images_alongside_it_is_not_reclassified(tmp_path):
    """A dataset card with screenshots is still a dataset."""
    (tmp_path / "train.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    _images(tmp_path / "figures", 9)
    assert detect_foreign_dataset(tmp_path) is None


def test_an_arbitrary_directory_is_not_guessed_at(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "more.txt").write_text("hello", encoding="utf-8")
    assert detect_foreign_dataset(tmp_path) is None


def test_a_handful_of_images_is_below_the_evidence_threshold(tmp_path):
    """Two images in two directories is not a dataset; it is a coincidence."""
    _images(tmp_path / "a", 1)
    _images(tmp_path / "b", 1)
    assert detect_foreign_dataset(tmp_path) is None


def test_one_directory_of_images_is_not_the_imagefolder_convention(tmp_path):
    """Without per-class directories there is no classification signature to read."""
    _images(tmp_path / "photos", 20)
    assert detect_foreign_dataset(tmp_path) is None


def test_an_empty_directory_is_not_a_vision_dataset(tmp_path):
    assert detect_foreign_dataset(tmp_path) is None


def test_a_file_path_is_not_walked(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("{}", encoding="utf-8")
    assert detect_foreign_dataset(path) is None


def test_images_with_unrelated_filenames_are_not_called_segmentation(tmp_path):
    """images/ + masks/ that do not correspond is not a segmentation pair.

    It still reads as classification -- two directories of images, no annotations -- which
    is the honest answer: it is a vision upload of some kind, and the layout does not say
    which. Naming the wrong vision task is the failure this correspondence check prevents.
    """
    _images(tmp_path / "images", 6, prefix="a")
    _images(tmp_path / "masks", 6, suffix=".png", prefix="totally-different")
    found = detect_foreign_dataset(tmp_path)
    assert found is not None
    assert found.task == TASK_IMAGE_CLASSIFICATION


def test_a_truncated_walk_does_not_hide_the_jsonl_that_forces_abstention(tmp_path, monkeypatch):
    """The bound is a performance guard; it must not become a correctness hole.

    With the walk cut short, "no .jsonl seen" is not "no .jsonl present". Here the images
    are found first and the .jsonl only after the cap, which without the confirmation step
    would tell an LM user their dataset belongs to a vision pipeline.
    """
    from lmpipeline.datasets import foreign

    _imagefolder(tmp_path)
    (tmp_path / "zzz-late").mkdir()
    (tmp_path / "zzz-late" / "train.jsonl").write_text('{"messages": []}\n', encoding="utf-8")

    # rglob walks breadth-first: 3 directories, then all 11 images, then the .jsonl. A cap
    # of 14 therefore sees the FULL classification signature and stops one entry short of
    # the evidence that must override it -- the only arrangement that tests the
    # confirmation step rather than the evidence threshold. Verified by disabling the
    # confirmation: without it this returns an image-classification identification.
    assert len(list(tmp_path.rglob("*"))) == 15
    monkeypatch.setattr(foreign, "MAX_ENTRIES_SCANNED", 14)
    assert foreign.detect_foreign_dataset(tmp_path) is None


def test_packaging_noise_is_ignored(tmp_path):
    """__MACOSX shadows every file in a Mac-made zip and would double every count."""
    _imagefolder(tmp_path)
    _images(tmp_path / "__MACOSX" / "cats", 6)
    found = detect_foreign_dataset(tmp_path)
    assert found.evidence["imageFiles"] == 11


# -- integration with split resolution -----------------------------------------------


def test_resolution_reports_the_wrong_pipeline_instead_of_a_missing_split(tmp_path):
    """The behaviour the issue actually asks for, at the boundary the user sees."""
    dataset = tmp_path / "dataset"
    _imagefolder(dataset)
    with pytest.raises(DatasetError) as excinfo:
        resolve_dataset(dataset, workdir=tmp_path / "work")
    assert excinfo.value.code == Code.DATASET_WRONG_PIPELINE
    assert TASK_IMAGE_CLASSIFICATION in excinfo.value.message
    assert excinfo.value.details["detectedTask"] == TASK_IMAGE_CLASSIFICATION


def test_resolution_still_reports_a_missing_split_for_an_lm_dataset(tmp_path):
    """The ordinary error must remain the default, not become the fallback."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(DatasetError) as excinfo:
        resolve_dataset(dataset, workdir=tmp_path / "work")
    assert excinfo.value.code == Code.DATASET_SPLIT_MISSING


def test_a_zipped_image_dataset_is_recognized_after_extraction(tmp_path):
    """Uploads arrive as archives, so detection must run on the extracted tree."""
    import zipfile

    staging = tmp_path / "staging"
    _imagefolder(staging)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with zipfile.ZipFile(dataset / "upload.zip", "w") as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    with pytest.raises(DatasetError) as excinfo:
        resolve_dataset(dataset, workdir=tmp_path / "work")
    assert excinfo.value.code == Code.DATASET_WRONG_PIPELINE


def test_the_wrong_pipeline_error_carries_no_file_contents(tmp_path):
    """Structure only: names and counts may travel, payload may not."""
    dataset = tmp_path / "dataset"
    _imagefolder(dataset)
    (dataset / "cats" / "secret.jpg").write_bytes(b"CONFIDENTIAL-PIXELS")
    with pytest.raises(DatasetError) as excinfo:
        resolve_dataset(dataset, workdir=tmp_path / "work")
    serialized = repr(excinfo.value.details) + excinfo.value.message
    assert "CONFIDENTIAL-PIXELS" not in serialized
    assert "secret.jpg" not in serialized

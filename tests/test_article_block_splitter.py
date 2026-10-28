import numpy as np
import pandas as pd
import pytest
from PIL import Image

from segment_article import ArticleBlockSplitter, article_block_splitter


def make_block(block_id="block"):
    return pd.Series(
        {
            "id": block_id,
            "hpos": 0,
            "vpos": 0,
            "width": 100,
            "height": 100,
        }
    )


@pytest.mark.parametrize("as_pil", [False, True])
def test_batch_without_detected_lines_copies_lines_and_preserves_blocks(
    monkeypatch, as_pil
):
    monkeypatch.setattr(
        article_block_splitter,
        "detect_horizontal_lines_in_block",
        lambda **kwargs: [],
    )
    lines_df = pd.DataFrame(
        {
            "block_id": ["first", "second"],
            "vpos": [10, 60],
            "height": [5, 5],
        }
    )
    blocks_df = pd.DataFrame(
        [make_block("first"), make_block("second")]
    )
    image_array = np.zeros((100, 100), dtype=np.uint8)
    image = Image.fromarray(image_array) if as_pil else image_array

    result = ArticleBlockSplitter().split_blocks(
        lines_df=lines_df,
        blocks_df=blocks_df,
        img=image,
    )

    assert result["lines_df"] is not lines_df
    assert result["lines_df"]["block_id"].tolist() == ["first", "second"]
    assert result["lines_df"]["original_block_id"].tolist() == ["first", "second"]
    assert "original_block_id" not in lines_df.columns
    assert result["blocks_df"]["id"].tolist() == ["first", "second"]
    assert result["blocks_df"]["original_id"].tolist() == ["first", "second"]


def test_single_block_split_assigns_lines_and_mutates_input(monkeypatch):
    monkeypatch.setattr(
        article_block_splitter,
        "detect_horizontal_lines_in_block",
        lambda **kwargs: [(10, 48, 90, 48)],
    )
    lines_df = pd.DataFrame(
        {
            "block_id": ["block", "block", "block"],
            "vpos": [10, 52, 80],
            "height": [4, 4, 4],
        }
    )

    result = ArticleBlockSplitter().split_block(
        df=lines_df,
        row=make_block(),
        img=np.zeros((100, 100), dtype=np.uint8),
    )

    assert result["lines_df"] is lines_df
    assert result["blocks_df"]["id"].tolist() == ["block_1", "block_2"]
    assert result["blocks_df"]["vpos"].tolist() == [0, 50]
    assert result["blocks_df"]["height"].tolist() == [46, 50]
    assert lines_df["block_id"].tolist() == ["block_1", "block_2", "block_2"]
    assert lines_df["original_block_id"].tolist() == ["block"] * 3


def test_splitter_passes_instance_detector_settings(monkeypatch):
    detected_options = {}

    def detect(**kwargs):
        detected_options.update(kwargs)
        return []

    monkeypatch.setattr(article_block_splitter, "detect_horizontal_lines_in_block", detect)
    splitter = ArticleBlockSplitter(
        min_line_width_ratio=0.25,
        max_line_gap=7,
        hough_threshold=23,
        center_tolerance=0.2,
    )

    splitter.split_block(
        df=pd.DataFrame({"block_id": ["block"]}),
        row=make_block(),
        img=np.zeros((100, 100), dtype=np.uint8),
    )

    assert detected_options["min_line_width_ratio"] == 0.25
    assert detected_options["max_line_gap"] == 7
    assert detected_options["hough_threshold"] == 23


def test_single_block_accepts_dict_row_without_split(monkeypatch):
    monkeypatch.setattr(
        article_block_splitter,
        "detect_horizontal_lines_in_block",
        lambda **kwargs: [],
    )
    lines_df = pd.DataFrame({"block_id": ["block"]})

    result = ArticleBlockSplitter().split_block(
        df=lines_df,
        row=make_block().to_dict(),
        img=np.zeros((100, 100), dtype=np.uint8),
    )

    assert result["blocks_df"].loc[0, "original_id"] == "block"
    assert lines_df.loc[0, "original_block_id"] == "block"


def test_center_tolerance_controls_candidate_line_filtering(monkeypatch):
    monkeypatch.setattr(
        article_block_splitter,
        "detect_horizontal_lines_in_block",
        lambda **kwargs: [(25, 48, 45, 48)],
    )
    lines_df = pd.DataFrame({"block_id": ["block"], "vpos": [60]})

    result = ArticleBlockSplitter(center_tolerance=0.2).split_block(
        df=lines_df,
        row=make_block(),
        img=np.zeros((100, 100), dtype=np.uint8),
    )

    assert result["blocks_df"]["id"].tolist() == ["block_1", "block_2"]


def test_line_on_shared_boundary_is_assigned_to_later_block():
    lines_df = pd.DataFrame(
        {
            "block_id": ["block"] * 3,
            "vpos": [10, 30, 50],
            "height": [5, 5, 5],
        }
    )
    blocks_df = pd.DataFrame(
        [
            {"id": "block_1", "vpos": 10, "height": 20},
            {"id": "block_2", "vpos": 30, "height": 20},
        ]
    )

    result = article_block_splitter.assign_lines_to_new_blocks(lines_df, blocks_df)

    assert result["block_id"].tolist() == ["block_1", "block_2", "block_2"]
    assert result["original_block_id"].tolist() == ["block"] * 3
    assert lines_df["block_id"].tolist() == ["block"] * 3
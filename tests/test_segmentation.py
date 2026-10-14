import pytest
from lxml import etree

from segment_article.segmentation import SegmentPage


ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"


def make_alto_page(previous_text, current_text, previous_width=100, current_width=100):
    root = etree.Element(f"{{{ALTO_NS}}}alto", nsmap={None: ALTO_NS})
    layout = etree.SubElement(root, f"{{{ALTO_NS}}}Layout")
    page = etree.SubElement(layout, f"{{{ALTO_NS}}}Page")
    print_space = etree.SubElement(page, f"{{{ALTO_NS}}}PrintSpace")
    block = etree.SubElement(
        print_space,
        f"{{{ALTO_NS}}}TextBlock",
        ID="block",
        HPOS="0",
        VPOS="0",
        WIDTH="100",
        HEIGHT="120",
    )

    for line_id, vpos, width, text in (
        ("line1", "0", previous_width, previous_text),
        ("line2", "100", current_width, current_text),
    ):
        line = etree.SubElement(
            block,
            f"{{{ALTO_NS}}}TextLine",
            ID=line_id,
            HPOS="0",
            VPOS=vpos,
            WIDTH=str(width),
            HEIGHT="10",
        )
        etree.SubElement(line, f"{{{ALTO_NS}}}String", CONTENT=text)

    return root


@pytest.mark.parametrize(
    ("previous_text", "current_text", "expected_start"),
    [
        ("short", "a much longer continuation", False),
        ("abcdefghi.", "klmnopqrst", True),
        ("abcdefghij", "Klmnopqrst", True),
    ],
)
def test_article_start_uses_adjacent_line_text(
    previous_text, current_text, expected_start
):
    page = SegmentPage(
        make_alto_page(previous_text, current_text),
        article_gap_outlier_cutoff=30,
        merge_single_line_article_to_next=False,
    )

    result = page.segment()

    assert bool(result.loc[1, "is_article_start"]) == expected_start


def test_article_start_uses_line_width_similarity():
    page = SegmentPage(
        make_alto_page(
            "abcdefghij",
            "klmnopqrst",
            previous_width=100,
            current_width=98,
        ),
        article_gap_outlier_cutoff=30,
        merge_single_line_article_to_next=False,
    )

    result = page.segment()

    assert bool(result.loc[1, "is_article_start"])
# ResegmentArticle

Tools for re-segmenting Delpher newspaper articles. Given an article ID, the package takes the
article's ALTO OCR pages and:

1. selects the text blocks that belong to the article from:
    - The issues mpeg21 metadata (you need an api-key for copyrighted newspapers younger than 140 years)
    - or from coordinate zones you supply, which can be obtained through the SRU API 
    (use the parameter `x-fields=zones`; this results in an extra field `<zones>` with the 
    zoning in the right format)
2. splits those lines into **articles** (using vertical spacing between lines),
3. splits each article into **paragraphs** (using indentation of the first line),
4. joins hyphenated words and returns one row of text per paragraph.

See [docs/how_it_works.md](docs/how_it_works.md) for a detailed description of the logic. A
runnable quick start is in [examples/quickstart.ipynb](examples/quickstart.ipynb).

## Scope and known limitations

Please read this before using the output.

- **Regular articles only, not advertisements.** The method assumes running text in columns.
  Advertisements have free-form layouts, and article splitting in particular gives meaningless
  results on them.
- **Article splitting works best on older newspapers** with long columns, a simple layout and
  clear headers. On more modern newspapers (complex layouts, many short blocks) the splitting is
  less reliable. A higher `article_split_threshold` helps there, but it will also assign splits
  where there are none (and a too high value will miss real splits).
- Paragraph splitting is very good in all newspapers where paragraphs start with an
  indentation.** It is based on indentation only, so text with block-style paragraphs (no
  indentation) will not be split into paragraphs.


## Install

From this directory, install in editable mode:

```bash
python -m pip install -e ".[test]"
```

This makes `segment_article` importable from notebooks and scripts elsewhere. Run the tests with
`pytest`. Python 3.13 or higher is required. For the example notebook also install `jupyter` and
`python-dotenv`.

## Quick start

```python
import os
from segment_article import RedefineArticle

ra = RedefineArticle(
    article_id="ddd:000022499:mpeg21:a0013",
    delpher_api_key=os.getenv("DELPHER_API_KEY"), # Can be None for non copyrighted issues
)

paragraphs = ra.process_article()   # DataFrame, one row per paragraph
print(ra.paragraphs_to_text(paragraphs))
```

The ALTO pages and the metadata (MPEG21/DIDL) are fetched from `https://resolver.kb.nl`, which
requires a Delpher API key for copyright protected newspaper issues. The `delpher_api_key` 
is sent with both metadata and ALTO requests.
If `alto_cache_dir` / `mpeg21_cache_dir` are given, these directories are checked, 
before downloading files. Downloaded files will be stored here as well.
expected filenames are the identifier with `:` replaced by `_`, for example
`ddd_000022389_mpeg21_p002_alto.xml` and `ddd_000022499_mpeg21.xml`.

## Main classes

| Class / function | Purpose |
| --- | --- |
| `RedefineArticle` | Entry point. Resolves which blocks belong to an article, runs `SegmentPage` for each page, combines the pages (articles continuing across a page break are linked), optionally filters on keywords. |
| `SegmentPage` | Works on one ALTO page. Extracts lines, detects article and paragraph starts, merges lines into paragraph texts. |
| `ArticleBlockSplitter` | Splits blocks at detected horizontal rules in a page image and assigns lines to the resulting blocks. |
| `plot_blocks`, `fetch_image` | Visual checks: draw lines/blocks coloured by article or paragraph on the page scan. |

## Important parameters

`RedefineArticle` takes its own parameters, and forwards any additional keyword arguments to
`SegmentPage` (so `article_split_threshold=1.3` can be passed directly to `RedefineArticle`).

### `RedefineArticle`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `article_id` | required | Delpher article ID, e.g. `ddd:000022499:mpeg21:a0013`. |
| `zones` | `None` | List of dicts `{"x", "y", "w", "h", "image"}` (image coordinates, `image` e.g. `DDD_000022499_003_access.jp2`) to select blocks instead of extracting the zoning blocks from the issue mpeg21-metadata file. |
| `keywords` | `None` | If given, only articles containing a keyword are returned (plus up to `max_paragraph_gap` paragraphs between hits, assuming the article splits were too agressive, and these articles probably form one single article). |
| `max_paragraph_gap` | `5` | Keyword filtering: paragraphs between two hits that are still kept. `None` or `0` keeps everything between first and last hit. |
| `add_zoning` | `False` | Also build `zoning_df` (new bounding boxes per article part); `zoning_to_dict()` converts it to the `zones` format. |
| `clean_up` | `True` | Drops helper and intermediate columns used internally from the output. |
| `delpher_api_key` | `None` | API key sent when accessing metadata and ALTO files, including copyrighted material. |
| `alto_cache_dir`, `mpeg21_cache_dir` | `None` | Local caches for ALTO and metadata. |

### `SegmentPage` (forwarded)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `article_split_threshold` | `1.7` | **Most important parameter.** Z-score of the vertical gap between two lines needed to start a new article. Lower = more splits, higher = fewer. Use higher values for modern newspapers, and expect false splits. |
| `ignore_blocks` | `True` | Segment articles on line spacing across the whole page, instead of per-block. If set to False, at least each block (in the same column) is treated as an hard article split, before finding other splits. |
| `article_gap_outlier_cutoff` | `30` | Vertical gaps (ALTO units/pixels) at or above this always start a new article and are excluded from the z-score. |
| `article_segmentation` / `paragraph_segmentation` | `True` | Switch either step off (a single article / paragraph id is assigned). |
| `article_single_line_split` | `True` | A block with one line is its own article (not applied with `ignore_blocks`). |
| `merge_single_line_article_to_next` | `True` | A one-line article (usually a header) is merged with the next article. |
| `paragraph_split_threshold_z` | `1` | Z-score of a line's indentation needed to start a paragraph. |
| `paragraph_split_threshold_r` | `0.02` | Minimum indentation as a fraction of block width. Both `_z` and `_r` must be met. |
| `paragraph_single_line_split` | `True` | A block with one line is its own paragraph. |
| `keep_styles` | `True` | Keep font/paragraph style columns. Set to `False` for speed and memory. |
| `delpher_api_key` | `None` | Optional API key sent when fetching ALTO XML directly. |

## Output

`process_article()` (and `get_page_paragraphs()`) return a DataFrame with one row per paragraph:

| Column | Meaning |
| --- | --- |
| `article_id` | `A001`, `A002`, ... Article number over the whole document. An article continuing on the next page keeps its id. |
| `paragraph_id` | `P001`, ... Paragraph number within the page. `P000` marks text continuing a paragraph from the previous page. |
| `text` | Paragraph text, hyphenated words rejoined. |
| `page_id` | Page identifier, e.g. `ddd:000022499:mpeg21:p003`. |
| `line_id` | List of ALTO line ids in the paragraph. |

With `clean_up=False` the columns `page_idx`, `page_article_id` (page-local article id) and
`block_article_id` are kept as well. `get_page_paragraphs()` returns all paragraphs without keyword
filtering; `process_article()` additionally applies keywords and, when keywords are used, merges
paragraphs split over a page break.

Per-page objects are available afterwards via `ra._segment_page_objects` (one `SegmentPage` per
page, with `.df` for the line-level result), and `ra.lines_df` for all lines.

## Tuning and debugging

- Plot the result on the page scan with `plot_blocks(fetch_image(page_id), df, color_column="article_id")`
  (see the example notebook) and adjust `article_split_threshold`.
- Line-level results (with `is_article_start`, `is_paragraph_start`, `hpos_z`, `hpos_pct_diff`) are
  in `ra._segment_page_objects[i].df`.
- Too many article splits: raise `article_split_threshold`. Missing splits: lower it.
- Paragraphs not found: check whether the paragraphs are indented at all; otherwise lower
  `paragraph_split_threshold_r`/`_z`, with the risk of false splits.

TODO: More tests.
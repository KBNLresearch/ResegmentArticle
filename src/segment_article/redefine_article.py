import ast
import re
from collections import OrderedDict
from itertools import pairwise
from pathlib import Path
import requests

import numpy as np
import pandas as pd
from lxml import etree

from .segmentation import SegmentPage


class RedefineArticle:
    """Retrieve or apply article zoning and re-segment its page content.

    The class obtains page and block selections from Delpher metadata or from
    caller-supplied image-coordinate zones, then uses :class:`SegmentPage` to
    create line and paragraph data. Processing can optionally retain zoning
    data and filter the resulting articles by keywords.

    Attributes:
        article_id: Delpher identifier of the article to process.
        paper_id: Paper identifier derived from ``article_id``.
        paragraph_df: Processed paragraph rows, including page-local and
            article-level identifiers.
        paragraph_texts: Paragraph rows returned by :meth:`process_article`.
        zoning_df: Processed text-block zoning data when ``add_zoning`` is true.
        lines_df: Concatenated, unfiltered line data from processed pages.
    """

    def __init__(
            self, 
            article_id, 
            keywords: list[str] | None = None,
            zones: list[dict] | str | None = None,
            max_paragraph_gap: int | None = 5,
            add_zoning: bool = False,
            xzwh_zoning: bool = True,
            clean_up=True,
            delpher_api_key: str | None = None,
            alto_cache_dir: str | Path | None = None,
            mpeg21_cache_dir: str | Path | None = None,
            **kwargs
        ):
        """Configure article retrieval, segmentation, and optional filtering.

        Args:
            article_id: Delpher identifier for the article, including its page
                or article suffix.
            keywords: Optional terms used to retain matching articles and
                paragraphs between keyword hits.
            zones: Optional list of image-coordinate zone dictionaries, or a
                string representation of that list. Each zone must include an
                ``image`` filename and coordinates accepted by ``SegmentPage``.
            max_paragraph_gap: Maximum number of intervening paragraphs to keep
                between keyword hits. A false value keeps all paragraphs between
                the first and last hit.
            add_zoning: Whether to build and expose text-block zoning data.
            xzwh_zoning: Whether :meth:`zoning_to_dict` uses ``x``, ``y``, ``w``,
                ``h`` coordinate keys by default.
            clean_up: Whether to remove intermediate columns from returned text.
            delpher_api_key: API key sent when retrieving Delpher metadata.
            alto_cache_dir: Optional directory used by ``SegmentPage`` for ALTO
                XML caching, or pre-downloaded ALTO XML files.
            mpeg21_cache_dir: Optional directory used for MPEG21 XML caching, or
                pre-downloaded MPEG21 XML files.
            **kwargs: Additional options forwarded to each ``SegmentPage``.

        Raises:
            ValueError: If ``zones`` is a string that does not evaluate to a
                list.
        """

        self.article_id = article_id
        self.paper_id = article_id.rsplit(":", 1)[0]
        self.keywords = keywords
        
        # Zoning from the server comes as a string
        if isinstance(zones, str):
            zones = ast.literal_eval(zones)
            if not isinstance(zones, list):
                raise ValueError("Zones must be a list")
            
        self.zones : list[dict] | None = zones
        self.page_zones: dict[str, list[dict]] = {}
        self._page_images: dict[str, str] = {}
        self.add_zoning = add_zoning
        self.max_paragraph_gap = max_paragraph_gap
        self.clean_up = clean_up
        self.xzwh_zoning = xzwh_zoning
        self.alto_cache_dir = Path(alto_cache_dir) if alto_cache_dir else None
        self.mpeg21_cache_dir = Path(mpeg21_cache_dir) if mpeg21_cache_dir else None
        self.segment_page_kwargs = kwargs
        self.delpher_api_key = delpher_api_key

        self.zoning : list[pd.DataFrame] = []
        self.page_block_filter : dict|None = None

    def __call__(self, **kwargs):
        """Process the article; equivalent to calling :meth:`process_article`.

        Args:
            **kwargs: Keyword arguments forwarded to :meth:`process_article`.

        Returns:
            pd.DataFrame: Processed paragraph rows.
        """
        return self.process_article(**kwargs)


    def process_article(
            self, 
            keywords: list[str] | None = None
        ):
        """Retrieve zoning, segment pages, and optionally filter by keywords.

        Args:
            keywords: Terms used to select relevant paragraphs and articles.
                If omitted, the value supplied to the constructor is used.

        Returns:
            pd.DataFrame: Paragraph text and associated metadata. When keyword
                filtering is enabled, paragraphs are merged across page breaks
                and labeled with the original ``article_id``.

        Side Effects:
            Updates ``page_block_filter``, ``paragraph_df``,
            ``paragraph_texts``, and the page-processing state. When zoning is
            enabled, also prepares ``zoning_df``.
        """

        keywords = self.keywords if keywords is None else keywords

        self.page_block_filter = self.get_article_zoning()
        self.paragraph_df = self.get_page_paragraphs(self.page_block_filter)

        if keywords:
            self.paragraph_df = self.merge_split_articles(keywords)
            self.paragraph_texts = SegmentPage.merge_paragraphs_over_page_split(self.paragraph_df)
            self.paragraph_texts['article_id'] = self.article_id
        else:
            self.paragraph_texts = self.paragraph_df.copy()

        if self.clean_up:
            self.paragraph_texts = self.paragraph_texts.drop(
                columns=['block_article_id', 'page_article_id', 'page_idx'], 
                errors='ignore'
            )

        return self.paragraph_texts.copy()
    

    def get_zoning_from_coordinates(self, zones : list[dict] | None = None) -> dict:
        """Group caller-supplied image-coordinate zones by page without DIDL.

        The page id is derived from the image filename, e.g.
        'DDD_000022499_003_access.jp2' -> 'ddd:000022499:mpeg21:p003'.
        Block ids are resolved per page by SegmentPage.
        
        Args:
            zones (list[dict] | None): List of coordinate zones, each represented as 
                a dictionary containing an "image" key with the image filename and
                "x", "y", "w", "h" keys for the coordinates.

        Returns:
            dict[str, list[str]]: Ordered mapping of page identifiers to empty
                block filters; the coordinate zones are stored in ``page_zones``
                for ``SegmentPage`` to resolve.

        Raises:
            ValueError: If a zone image filename does not contain a page number
                followed by ``_access``.
        """

        zones = zones if zones is not None else self.zones
        if not zones:
            raise ValueError("No zones provided as argument or in self.zones.")

        zone_keys = {"image", "x", "y", "w", "h"}

        self.page_zones = OrderedDict()
        for zone in zones:
            if not zone:
                continue

            # Validate zone dictionary
            missing_keys = zone_keys - set(zone.keys())
            if missing_keys:
                raise ValueError(f"Invalid zone dictionary; missing keys: {missing_keys}")
            
            match = re.search(r"_(\d+)_access", zone["image"])
            if not match:
                raise ValueError(f"Cannot derive page number from {zone['image']}")
            page_id = f"{self.paper_id}:p{match.group(1)}"
            self.page_zones.setdefault(page_id, []).append(zone)
            self._page_images[page_id] = zone["image"]

        return OrderedDict((page_id, []) for page_id in self.page_zones)


    def get_article_zoning(self) -> dict:
        """Get the article's page-to-block selection.

        If coordinate zones were supplied, groups those zones by page. Otherwise
        retrieves the paper's DIDL metadata and extracts the text block IDs for
        the article's parts.

        Returns:
            dict[str, list[str]]: Ordered mapping from page identifiers to the
                selected ALTO text-block IDs. For coordinate zones, values are
                empty lists because blocks are resolved from the zones later.
        """

        if self.zones:
            return self.get_zoning_from_coordinates()

        # Extract paper identifier from article identifier
        paper_identifier = self.article_id.rsplit(":", 1)[0]

        self.didl = self.load_mpeg_metadata(paper_identifier)

        # Get didl element (didl:DIDL) with the correct namespaces
        self.didl = self.didl_tree.find(".//{urn:mpeg:mpeg21:2002:02-DIDL-NS}DIDL")

        nsmap = self.didl.nsmap
        # Find Component with identifier <article_identifier>:zoning
        zoning_id = f"{self.article_id}:zoning"
        zoning_component = self.didl.find(
            f'.//didl:Component[@dc:identifier="{zoning_id}"]',
            namespaces=nsmap
        )

        # Find dcx:article-part elements
        article_parts = zoning_component.findall(
            ".//dcx:article-part", 
            namespaces=nsmap
        )

        block_data = OrderedDict()
        for part in article_parts:
            text_blocks = part.findall(
                ".//dcx:blocks/dcx:TextBlock", 
                namespaces=nsmap
            )
            block_ids = [block.attrib['ID'] for block in text_blocks]

            block_data[part.attrib['pageid']] = block_ids

        if len(block_data) == 0:
            raise ValueError("No text blocks found in the article parts. Please check the article identifier and the MPEG metadata.")

        return block_data


    def load_mpeg_metadata(
            self,
            paper_identifier: str,
    ) -> etree._Element:
        """Load MPEG metadata for the given paper identifier. Either from cache,
        or by fetching it from the remote resolver if not cached.

        Args:
            paper_identifier: The identifier of the paper to load metadata for.

        Returns:
            etree._Element: The root element of the MPEG metadata XML tree.
        """
        use_cache = bool(self.mpeg21_cache_dir)

        if use_cache:
            filename = paper_identifier.replace(":", "_") + ".xml"
            cache_file = self.mpeg21_cache_dir / filename

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    didl_content = f.read()
                self.didl_tree = etree.fromstring(didl_content)
                return self.didl_tree

        # Get metadata for current paper
        self.url = f"https://resolver.kb.nl/resolve?urn={paper_identifier}"
        
        headers = {}
        if self.delpher_api_key:
            headers = {
                "x-api-key" : self.delpher_api_key
            }

        with requests.get(self.url, headers=headers) as response:
            response.raise_for_status()
            didl_content = response.content
            self.didl_tree = etree.fromstring(didl_content)

            if use_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    f.write(didl_content)

        return self.didl_tree

        
    def get_page_paragraphs(
            self,
            page_block_filter : dict|None = None,
        ) -> pd.DataFrame:
        """Segment selected blocks on each page and combine their paragraphs.

        Args:
            page_block_filter: Optional mapping from page identifiers to lists
                of ALTO block IDs. If omitted or false-valued, uses the current
                filter or obtains it with :meth:`get_article_zoning`.

        Returns:
            pd.DataFrame: Combined paragraph rows with page identifiers and
                article IDs reassigned across page boundaries.

        Side Effects:
            Populates the line and paragraph dataframes and the per-page
            ``SegmentPage`` objects. If ``add_zoning`` is true, also populates
            ``zoning`` and ``zoning_df``.
        """

        if not page_block_filter:
            if not self.page_block_filter:
                self.page_block_filter = self.get_article_zoning()

            page_block_filter = self.page_block_filter


        self.lines_dfs = []
        self._segment_page_objects = []
        self.paragraph_dfs = []
        self.zoning = []

        for idx, (page_id, block_ids) in enumerate(page_block_filter.items()):
            self._page_id = page_id
            self._alto_url = f"https://resolver.kb.nl/resolve?urn={page_id}:alto"

            # Create new segmentation for selected blocks on the current page
            self._segment_page = SegmentPage(
                alto_xml_tree=self._alto_url,
                block_filter=block_ids,
                zone_filter=self.page_zones.get(page_id),
                clean_up=self.clean_up,
                alto_cache_dir=self.alto_cache_dir,
                **self.segment_page_kwargs
            )
            lines_df = self._segment_page.segment()
            self._segment_page_objects.append(self._segment_page)

            lines_df['page_id'] = page_id
            lines_df['page_idx'] = idx
            self.lines_dfs.append(lines_df)

            df = self._segment_page.merge_paragraphs(
                lines_df, 
            )
            df['page_id'] = page_id
            df['page_idx'] = idx

            if self.add_zoning:
                self.zoning.append(self.get_page_zoning(self._segment_page, page_id))

            self.paragraph_dfs.append(df)

        self._paragraph_df = self.assign_article_ids(
            pd.concat(self.paragraph_dfs, ignore_index=True)
        )
        self.paragraph_df = self._paragraph_df.copy()

        if self.add_zoning:
            self._zoning_df = self.assign_zoning_article_ids(
                pd.concat(self.zoning, ignore_index=True),
                self._paragraph_df,
            )
            self.zoning_df = self._zoning_df.copy()

        return self.paragraph_df


    @staticmethod
    def assign_article_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Reassign article IDs across pages while retaining page-local IDs.

        The page-local ID is kept in
        'page_article_id'; the per-page dataframes (lines_dfs, paragraph_dfs)
        are not modified and keep the page-local ids.

        The first paragraph of a page that does not start with a paragraph break
        (paragraph_id P000) continues the last article of the previous page.

        Args:
            df: Paragraph rows containing ``page_idx``, ``article_id``, and
                ``paragraph_id`` columns.

        Returns:
            pd.DataFrame: A copy with the original IDs renamed to
                ``page_article_id`` and new cross-page ``article_id`` values.
        """
        df = df.rename(columns={'article_id': 'page_article_id'})

        new_page = df['page_idx'] != df['page_idx'].shift()
        new_page_article = df['page_article_id'] != df['page_article_id'].shift()
        continues = new_page & (df['paragraph_id'].str[1:].astype(int) == 0)
        continues.iloc[0] = False

        is_new = (new_page | new_page_article) & ~continues
        df['article_id'] = 'A' + is_new.cumsum().astype(str).str.zfill(3)

        return df


    @staticmethod
    def assign_zoning_article_ids(
            zoning_df: pd.DataFrame,
            paragraph_df: pd.DataFrame,
        ) -> pd.DataFrame:
        """Attach reassigned paragraph article IDs to their zoning blocks.

        The page-local article ID is retained as ``page_article_id``.

        Args:
            zoning_df: Zoning rows containing ``page_id``, ``block_article_id``,
                and ``article_id`` columns.
            paragraph_df: Paragraph rows containing ``page_id``,
                ``block_article_id``, and reassigned ``article_id`` columns.

        Returns:
            pd.DataFrame: Zoning rows joined to the reassigned article IDs by
                page and block. Unmatched zoning rows have a missing article ID.
        """
        zoning_df = zoning_df.rename(columns={'article_id': 'page_article_id'})

        block_articles = paragraph_df\
            .explode('block_article_id')[['page_id', 'block_article_id', 'article_id']]\
            .drop_duplicates()

        return zoning_df.merge(block_articles, on=['page_id', 'block_article_id'], how='left')


    def merge_split_articles(self, keywords: list[str]) -> pd.DataFrame:
        """Keep articles selected by keyword hits and the configured gap rule.

        Matching paragraphs select their full reassigned article. Paragraphs
        between successive hits are included when their count does not exceed
        ``max_paragraph_gap``; a false value includes the full span between the
        first and last hit. Article IDs already account for page continuations.

        If zoning is enabled, ``zoning_df`` is filtered to the selected articles.

        Args:
            keywords: Terms to search for in paragraph text, matched without
                regard to case.

        Returns:
            pd.DataFrame: Selected paragraph rows in their original order.
        """
        df = self._paragraph_df.reset_index(drop=True)

        pattern = '|'.join(re.escape(keyword.lower()) for keyword in keywords)
        mask = df['text'].str.contains(pattern, case=False, na=False).to_numpy().copy()

        if mask.any():
            hits = np.flatnonzero(mask)
            if self.max_paragraph_gap:
                for left, right in pairwise(hits):
                    if 0 < right - left - 1 <= self.max_paragraph_gap:
                        mask[left + 1:right] = True
            else:
                mask[hits[0]:hits[-1] + 1] = True

        # article_ids already span page breaks (see assign_article_ids)
        relevant = set(df.loc[mask, 'article_id'])
        is_relevant = df['article_id'].isin(relevant)

        if self.add_zoning:
            self.zoning_df = self._zoning_df[self._zoning_df['article_id'].isin(relevant)]\
                .reset_index(drop=True)

        return df[is_relevant].reset_index(drop=True)


    def get_page_zoning(
            self,
            segment_page_obj : SegmentPage,
            page_id: str,
        ) -> pd.DataFrame:
        """Build zoning rows for a segmented page.

        Args:
            segment_page_obj: Segmentation object containing the page's newly
                created text blocks.
            page_id: Identifier of the page represented by the object.

        Returns:
            pd.DataFrame: Text-block zoning rows augmented with ``page_id`` and
                the page image filename.
        """

        block_df = segment_page_obj.create_new_text_blocks_df()
        block_df['page_id'] = page_id
        block_df['image'] = self.get_page_image_filename(page_id)

        return block_df


    def zoning_to_dict(
            self,
            zoning_df: pd.DataFrame | None = None,
            use_xzwh: bool | None = None,
        ) -> list[dict]:
        """
        Convert the zoning dataframe to a list of dictionaries, with the coordinates
        and image filename. Coordinates use either ``hpos``, ``vpos``, ``width``,
        and ``height`` or ``x``, ``y``, ``w``, and ``h``, depending on
        ``use_xzwh``.

        Args:
            zoning_df: Zoning rows to convert. If ``None``, uses ``self.zoning_df``.
            use_xzwh: Whether to use ``x``, ``y``, ``w``, ``h`` coordinate keys.
                If ``None``, uses ``self.xzwh_zoning``.

        Returns:
            list[dict]: One dictionary per zoning row, containing the selected
                coordinate keys and the ``image`` filename.
        """

        if isinstance(zoning_df, type(None)):
            zoning_df = self.zoning_df

        if use_xzwh is None:
            use_xzwh = self.xzwh_zoning

        column_mapping = {
            'hpos': 'x',
            'vpos': 'y',
            'width': 'w',
            'height': 'h'
        }

        if use_xzwh:
            zoning_df = zoning_df.copy().rename(columns=column_mapping)

        columns_to_keep = list(column_mapping.keys()) if not use_xzwh else list(column_mapping.values())
        columns_to_keep += ['image']

        # Create a list with dictionaries
        output_dict = zoning_df[columns_to_keep]\
            .to_dict(orient='records')

        return output_dict


    @property
    def lines_df(self) -> pd.DataFrame:
        """Return all processed line rows, before paragraph/article filtering.

        Returns:
            pd.DataFrame: Concatenated line data from every processed page. The
                result is cached after the first access.
        """
        if not hasattr(self, '_lines_df'):
            self._lines_df = pd.concat(self.lines_dfs, ignore_index=True)
        return self._lines_df


    def get_page_image_filename(
            self, 
            page_id: str
        ):
        """Get a page image filename from supplied zones or DIDL metadata.

        Args:
            page_id: Identifier of the page whose image filename is requested.

        Returns:
            str: Image filename associated with the page.
        """
        if page_id in self._page_images:
            return self._page_images[page_id]

        nsmap = self.didl.nsmap

        image_resource = self.didl.find(
            f'.//didl:Component[@dc:identifier="{page_id}:image"]/didl:Resource',
            namespaces=nsmap
        )

        filename_key = f"{{{nsmap['dcx']}}}filename"
        return image_resource.attrib[filename_key]

    
    def paragraphs_to_text(self, texts_df, sep="\n\n") -> str:
        """Join paragraph text values into a single string.

        Args:
            texts_df: DataFrame containing a ``text`` column.
            sep: String inserted between consecutive paragraph texts.

        Returns:
            str: Joined paragraph text.
        """
        return sep.join(texts_df['text'].tolist())


    def paragraphs_to_xml(
            self,
            df: pd.DataFrame | None = None
        ) -> str:
        """Wrap paragraph texts in a simple ``<text>`` XML document, to the same
        format as the source OCR XML data in the newspaper collection.

        Args:
            df: Paragraph DataFrame containing a ``text`` column. If ``None``,
                uses ``self.paragraph_texts``.

        Returns:
            str: XML declaration, root element, and one ``<p>`` element per row.

        Note:
            Text values are inserted as-is and are not XML-escaped.
        """
        if isinstance(df, type(None)):
            df = self.paragraph_texts

        p = "<p>" + df['text'] + "</p>"

        elements = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<text>",
            *p.tolist(),
            "</text>"
        ]

        return "\n".join(elements)
import ast
import re
from collections import OrderedDict
from itertools import pairwise
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from lxml import etree

from .segmentation import SegmentPage


class RedefineArticle:
    def __init__(
            self, 
            article_id, 
            keywords: list[str] | None = None,
            zones: list[dict] | str | None = None,
            max_paragraph_gap: int | None = 5,
            add_zoning: bool = False,
            xzwh_zoning: bool = True,
            clean_up=True,
            alto_cache_dir: str | Path | None = None,
            **kwargs
        ):

        self.article_id = article_id
        self.paper_id = article_id.rsplit(":", 1)[0]
        self.keywords = keywords
        
        # Zoning from the server comes as a string
        if isinstance(zones, str):
            zones = ast.literal_eval(zones)
            if not isinstance(zones, list):
                raise ValueError("Zones must be a list of dictionaries")
            
        self.zones : list[dict] | None = zones
        self.page_zones: dict[str, list[dict]] = {}
        self._page_images: dict[str, str] = {}
        self.add_zoning = add_zoning
        self.max_paragraph_gap = max_paragraph_gap
        self.clean_up = clean_up
        self.xzwh_zoning = xzwh_zoning
        self.segment_page_kwargs = kwargs
        self.alto_cache_dir = alto_cache_dir

        self.zoning : list[pd.DataFrame] = []
        self.page_block_filter : dict|None = None

    def __call__(self, **kwargs):
        return self.process_article(**kwargs)


    def process_article(
            self, 
            keywords: list[str] | None = None
        ):

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
    

    def get_zoning_from_coordinates(self) -> dict:
        """
        Group the user-supplied zones per page, without requesting the DIDL.
        The page id is derived from the image filename, e.g.
        'DDD_000022499_003_access.jp2' -> 'ddd:000022499:mpeg21:p003'.
        Block ids are resolved per page by SegmentPage.
        """
        self.page_zones = OrderedDict()
        for zone in self.zones:
            if not zone or not zone.get("image"):
                continue
            match = re.search(r"_(\d+)_access", zone["image"])
            if not match:
                raise ValueError(f"Cannot derive page number from {zone['image']}")
            page_id = f"{self.paper_id}:p{match.group(1)}"
            self.page_zones.setdefault(page_id, []).append(zone)
            self._page_images[page_id] = zone["image"]

        return OrderedDict((page_id, []) for page_id in self.page_zones)


    def get_article_zoning(self) -> dict:

        if self.zones:
            return self.get_zoning_from_coordinates()

        # Extract paper identifier from article identifier
        paper_identifier = self.article_id.rsplit(":", 1)[0]

        # Get metadata for current paper
        self.url = f"https://resolver.kb.nl/resolve?urn={paper_identifier}"

        with requests.get(self.url) as response:
            response.raise_for_status()
            didl_content = response.content
            self.didl_tree = etree.fromstring(didl_content)

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
            print("No text blocks found in the article parts.")
        if len(block_data) > 2:
            print("Warning: More than 2 pages found in the article parts. Only the first 2 pages will be processed.")
            
        return block_data


    def get_page_paragraphs(
            self,
            page_block_filter : dict|None = None,
        ) -> pd.DataFrame:

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
        """
        Reassign article ids over all pages. The page-local id is kept in
        'page_article_id'; the per-page dataframes (lines_dfs, paragraph_dfs)
        are not modified and keep the page-local ids.

        The first paragraph of a page that does not start with a paragraph break
        (paragraph_id P000) continues the last article of the previous page.
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
        """Add the reassigned article_id to the zoning blocks, keeping the page-local one."""
        zoning_df = zoning_df.rename(columns={'article_id': 'page_article_id'})

        block_articles = paragraph_df\
            .explode('block_article_id')[['page_id', 'block_article_id', 'article_id']]\
            .drop_duplicates()

        return zoning_df.merge(block_articles, on=['page_id', 'block_article_id'], how='left')


    def merge_split_articles(self, keywords: list[str]) -> pd.DataFrame:
        """
        Keep only the (re)segmented articles that contain a keyword, plus the
        articles bridging the gaps between hits and continuing over a page break.

        Also filters self.zoning_df (if add_zoning) to the blocks of those articles.
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
        and the image filename. The coordinates can be in either be in ('hpos', 'vpos', 'width',
        'height') or (x, y, w, h) format, depending on the use_xzwh parameter.

        Args:
            zoning_df (pd.DataFrame | None): The zoning dataframe to convert. 
                If None, use self.zoning_df.
            use_xzwh (bool | None): Whether to use x, y, w, h format for the coordinates. 
                If None, use self.xzwh_zoning.

        Returns:
            list[dict]: A list of dictionaries, each containing the coordinates and the image filename.
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
        """
        Return the unfiltered lines dataframe of all the processed blocks.
        """
        if not hasattr(self, '_lines_df'):
            self._lines_df = pd.concat(self.lines_dfs, ignore_index=True)
        return self._lines_df


    def get_page_image_filename(
            self, 
            page_id: str
        ):
        """
        Get the filename of the page image from the zones or the DIDL XML.
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
        """
        Merge the texts of the paragraphs into a single text.
        """
        return sep.join(texts_df['text'].tolist())


    def paragraphs_to_xml(
            self,
            df: pd.DataFrame | None = None
        ) -> str:
        """
        Convert the paragraphs dataframe to XML format.
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
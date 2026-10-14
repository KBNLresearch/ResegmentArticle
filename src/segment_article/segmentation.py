import re
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import pandas as pd
import numpy as np
import requests
import validators
from lxml import etree
from .helpers import zscore
# from shapely.geometry import LineString

class SegmentPage:
    """
    Segment ALTO page content into article and paragraph-level units, using
    heuristics based on vertical and horizontal spacing, as well as logical flow
    between text blocks. The segmentation process is performed in three steps:
    1. Extract text lines from the ALTO XML and store them in a DataFrame
    2. Identify article boundaries based on vertical spacing.
    3. Identify paragraph boundaries based on horizontal spacing.
    
    """

    def __init__(
            self,
            alto_xml_tree: etree._Element | str | Path,
            block_filter: Sequence[str] | None = None,
            zone_filter: Sequence[Mapping[str, Any]] | None = None,
            paragraph_segmentation: bool = True,
            article_segmentation: bool = True,
            article_id_output_col: str = 'article_id',
            paragraph_id_output_col: str = 'paragraph_id',
            block_id_col: str = 'block_id',
            block_article_id_col: str = 'block_article_id',
            article_split_threshold: float = 1.7,
            article_single_line_split: bool = True,
            merge_single_line_article_to_next: bool = True,
            ignore_blocks: bool = False,
            article_gap_outlier_cutoff: float = 30,
            paragraph_split_threshold_z: float = 1,
            paragraph_split_threshold_r: float = 0.02,
            paragraph_single_line_split: bool = True,
            keep_styles: bool = True,
            clean_up: bool = True,
            use_alto_cache: bool = False,
            alto_cache_dir: str | Path | None=None,
        ) -> None:
        """
        Initialize the segmentation object from an ALTO XML source.

        Args:
            alto_xml_tree: ALTO XML content as an lxml element, file path, a page 
                identifier, or URL. 
            block_filter: Optional list of block IDs to restrict processing to.
            zone_filter: Optional list of zone coordinates (x, y, w, h or hpos, vpos,
                width, height), which are resolved to block IDs and override block_filter.
            paragraph_segmentation: Whether to segment paragraphs.
            article_segmentation: Whether to segment articles.
            article_id_output_col: Name of the column used for article identifiers.
            paragraph_id_output_col: Name of the column used for paragraph identifiers.
            block_id_col: Name of the column that stores block identifiers.
            block_article_id_col: Name of the column that stores combined block and 
                article identifiers.
            article_split_threshold: Threshold used to detect article boundaries.
            article_single_line_split: Whether a single line should trigger a new article.
            merge_single_line_article_to_next: Whether a single-line article should be 
                merged with the next article. Usually these lines are headers or titles.
            article_gap_outlier_cutoff: Vertical gaps above this value (ALTO units) always start
                a new article and are excluded from the gap z-score.
            ignore_blocks: Segment articles on line spacing across the whole page, ignoring
                block boundaries (also disables the logical flow, same column and single
                line rules, which are block-based).
            paragraph_split_threshold_z: Z-score threshold used to detect paragraph start 
                positions (z-score is based on relative horizontal distance of a line 
                to the text block boundary). Used in combination with `paragraph_split_threshold_r`,
                to prevent false positives due to low horizontal distance variations.
            paragraph_split_threshold_r: Ratio threshold (0-1) used to detect paragraph start 
                positions (on relative horizontal distance of a line to the text block boundary).
                Used in combination with `paragraph_split_threshold_z`, to prevent false positives 
                due to low horizontal distance variations.
            paragraph_single_line_split: Whether a block containing a single line should trigger a 
                new paragraph. Single blocks could be headers or titles.
            keep_styles: Whether to retain text and paragraph style information. Doing this
                can be useful for further analysis or processing of the segmented text.
                However, this causes overhead in terms of memory and processing time.
            clean_up: Whether to apply cleanup steps during segmentation.
            use_alto_cache: Whether to check for an existing ALTO XML file, if a url
                or page identifier is provided, and use it if available. If not 
                available, the ALTO XML will be downloaded and cached for future use.
            alto_cache_dir: Directory path to store or retrieve cached ALTO XML files. This is used
                in combination with `use_alto_cache`.
        """
        self.alto_xml_input = alto_xml_tree
        self.block_filter = list(block_filter or [])
        self.paragraph_segmentation = paragraph_segmentation
        self.article_segmentation = article_segmentation
        self.article_id_output_col = article_id_output_col
        self.paragraph_id_output_col = paragraph_id_output_col
        self.block_id_col = block_id_col
        self.block_article_id_col = block_article_id_col
        self.article_split_threshold = article_split_threshold
        self.article_single_line_split = article_single_line_split
        self.merge_single_line_article_to_next = merge_single_line_article_to_next
        self.ignore_blocks = ignore_blocks
        self.article_gap_outlier_cutoff = article_gap_outlier_cutoff
        self.paragraph_split_threshold_z = paragraph_split_threshold_z
        self.paragraph_split_threshold_r = paragraph_split_threshold_r
        self.paragraph_single_line_split = paragraph_single_line_split
        self.keep_styles = keep_styles
        self.clean_up = clean_up
        self.use_alto_cache = use_alto_cache
        self.alto_cache_dir = Path(alto_cache_dir) if alto_cache_dir is not None else None

        self.alto_xml_tree = self.load_alto_xml(alto_xml_tree)
        self.nsmap = self.alto_xml_tree.nsmap
        self.text_styles_df = self.get_text_styles_df() if self.keep_styles else pd.DataFrame()
        self.par_styles_df = self.get_paragraph_styles_df() if self.keep_styles else pd.DataFrame()

        self.zone_filter = list(zone_filter or [])
        if self.zone_filter:
            self.block_filter = self.zones_to_block_ids(self.zone_filter)

        self.text_blocks_df = pd.DataFrame()

        self.df: pd.DataFrame | None = None
        self.text_lines_df: pd.DataFrame | None = None
        self.article_df: pd.DataFrame | None = None
        self.paragraph_df: pd.DataFrame | None = None
        self.new_text_blocks_df: pd.DataFrame | None = None


    def __call__(self, block_filter: Sequence[str] | None = None) -> pd.DataFrame:
        return self.segment(block_filter=block_filter)


    def segment(
            self, 
            block_filter: Sequence[str] | None = None
        ) -> pd.DataFrame:
        """
        Segment the ALTO XML content into articles and paragraphs.

        Args:
            block_filter: Optional list of block IDs to restrict processing to.

        Returns:
            pd.DataFrame: A DataFrame containing the segmented text lines with article 
                and paragraph identifiers.
        """

        self.block_filter = self.block_filter if block_filter is None else list(block_filter)

        self.df = self.get_text_lines_df(block_filter=self.block_filter, keep_styles=self.keep_styles)
        
        if self.article_segmentation:
            self.df = self.segment_articles(self.df, block_filter=self.block_filter)
        else:
            # For compatibility with code expecting article segmentation
            self.df['is_article_start'] = False
            self.df.loc[self.df.index[0], 'is_article_start'] = True
            self.df[self.article_id_output_col] = "A001"
            self.df['block_article_id'] = self.df[self.block_id_col] + "_" + self.df[self.article_id_output_col]

        if self.paragraph_segmentation:
            self.df = self.segment_paragraphs(self.df, block_filter=self.block_filter)
        else:
            self.df[self.paragraph_id_output_col] = "P001"

        return self.df


    def zones_to_block_ids(self, zones: Sequence[Mapping[str, Any]]) -> list[str]:
        """
        Match zone coordinates exactly to TextBlock IDs.

        Raises:
            ValueError: If one or more zones do not match a TextBlock.
        """
        coord_cols = ["hpos", "vpos", "width", "height"]
        rename = {"x": "hpos", "y": "vpos", "w": "width", "h": "height"}

        zones_df = (
            pd.DataFrame(list(zones))
            .rename(columns=rename)
            .dropna(subset=coord_cols)
            .loc[:, coord_cols]
            .round()
            .astype(int)
            .drop_duplicates()
        )

        blocks = self.get_text_blocks_df()[["id", *coord_cols]].copy()
        blocks[coord_cols] = blocks[coord_cols].round().astype(int)

        merged = zones_df.merge(blocks, on=coord_cols, how="left")

        unmatched = merged[merged["id"].isna()]
        if not unmatched.empty:
            raise ValueError(
                f"{len(unmatched)} zone(s) do not match a TextBlock: "
                f"{unmatched[coord_cols].to_dict('records')}"
            )

        return merged["id"].tolist()


    def resegement_blocks(self):
        "Visual resegmentation of blocks"
        pass


    @staticmethod
    def _group_zscore(
        df: pd.DataFrame,
        values: pd.Series,
        group_columns: Sequence[str],
    ) -> pd.Series:
        work = df[list(group_columns)].copy()
        work["_value"] = values

        grouped = work.groupby(list(group_columns), sort=False)["_value"]
        mean = grouped.transform("mean")
        std = grouped.transform("std", ddof=0)

        return ((values - mean) / std).where(std.ne(0))
    
    def segment_articles(
            self, 
            df,
            block_filter: Sequence[str] | None = None
        ) -> pd.DataFrame:
        """
        Segment the text lines into articles based on vertical spacing.

        Spacing must be greater than the threshold on the z-score of the
        difference, and, as a guardrail, greater than a third of the median line 
        height within the block.

        Args:
            df (pd.DataFrame): DataFrame containing text lines with block identifiers.
            block_filter: Optional list of block IDs to restrict processing to.

        Returns:
            pd.DataFrame: A DataFrame containing the text lines with article identifiers.
        """
        # Filter the text blocks based on the provided block filter.
        block_filter = self.block_filter if block_filter is None else list(block_filter)
        self.text_blocks_df = self.get_text_blocks_df(block_filter=block_filter)

        # Group by block (if not ignoring blocks)
        group_column = self.block_id_col

        if self.ignore_blocks:
            # Create psudogroup for all lines when ignoring blocks
            group_column = "_page_group"
            df[group_column] = 0  

        grouped = df.groupby(group_column, sort=False)
        group_order = grouped.ngroup()

        # Calculate the vertical position differences between lines
        line_top = df["vpos"] + df["height"]
        previous_top = line_top.groupby(df[group_column], sort=False).shift()
        vpos_diff = df["vpos"] - previous_top

        # Large gaps are outliers and would mask smaller ones in the z-score.
        outliers = vpos_diff >= self.article_gap_outlier_cutoff
        z_input = vpos_diff.where(~outliers)
        if self.ignore_blocks:
            # Negative gaps are column jumps.
            z_input = z_input.where(z_input > 0)

        vpos_diff_z = self._group_zscore(
            df=df, 
            values=z_input, 
            group_columns=[group_column]
        )

        # Determine the minimum vertical difference required to consider a new article.
        min_difference = grouped["height"].transform("median") // 3

        # df["vpos_diff"] = vpos_diff
        # df["vpos_diff_z"] = vpos_diff_z

        df["is_article_start"] = (
            (vpos_diff_z >= self.article_split_threshold)
            & (vpos_diff >= min_difference)
        ) | outliers

        # Lines with 'hyppart2', (second part of a hyphenated word) cannot be a start of a new article.
        if 'hyppart2' in df.columns:
            df.loc[df['hyppart2'].notna(), "is_article_start"] = False

        if self.article_single_line_split and not self.ignore_blocks:
            single_line = grouped[group_column].transform("size").eq(1)
            df.loc[single_line, "is_article_start"] = True
        

        # Match groupby.apply's group order before assigning cumulative IDs.
        df = df.assign(_group_order=group_order)
        df = df.sort_values("_group_order", kind="stable").drop(columns="_group_order")
        df = df.reset_index(drop=True)

        # Define article start points
        if self.ignore_blocks:
            df = df.drop(columns=group_column)
            # The first line always opens an article.
            df.loc[0, "is_article_start"] = True
        else:
            text_blocks_df = self.text_blocks_df.set_index("id")
            logical_flow = df[self.block_id_col].map(text_blocks_df["logical_flow"])
            # First line of a block is considered an article start
            first_line = ~df[self.block_id_col].duplicated(keep="first")
            df.loc[~logical_flow & first_line, "is_article_start"] = True

            same_column = df[self.block_id_col].map(text_blocks_df["same_column"])
            df.loc[same_column & first_line, "is_article_start"] = True

        # Refine article split based on sentence endings and line widths
        if "text" in df.columns and len(df) > 1:
            df = self.filter_article_splits_based_on_capitalization_and_line_width(df)

        if self.merge_single_line_article_to_next:
            starts = df["is_article_start"].to_numpy(copy=True)
            # Sequential, so a merged article is not treated as single-line again.
            article_len = 0
            for i in range(len(starts)):
                if starts[i] and i > 0 and article_len == 1:
                    starts[i] = False
                if starts[i]:
                    article_len = 0
                article_len += 1
            df["is_article_start"] = starts

        df[self.article_id_output_col] = (
            "A" + df["is_article_start"].cumsum().astype(str).str.zfill(3)
        )
        df[self.block_article_id_col] = (
            df[self.block_id_col] + "_" + df[self.article_id_output_col]
        )

        self.article_df = df.copy()
        return df

    def filter_article_splits_based_on_capitalization_and_line_width(
            self,
            df,
    ) -> pd.DataFrame:
        """
        Filter article splits based on capitalization rules to avoid incorrect splits.
        
        For article splits, this method sets the "is_article_start" flag to False if:
            - The previous line ends with a sentence-ending punctuation  
            - And the current line does not start with a capital letter, indicating a 
                continuation of the same paragraph.
            - And the line widths of the previous and current lines are almost equal, 
                indicating that the lines likely belong to the same paragraph.
            All conditions must be met for the article start flag to be set to False.

        Args:
            df (pd.DataFrame): DataFrame containing text lines with article identifiers.

        Returns:
            pd.DataFrame: A DataFrame with potentially updated article start flags.

        """
        sub_df = df.copy()
        # Get lines with article start, and their predecessor line
        sub_df = sub_df[sub_df['is_article_start'] | sub_df['is_article_start'].shift(fill_value=False)]
        
        previous_text = sub_df["text"].shift().fillna("").astype(str)
        current_text = sub_df["text"].fillna("").astype(str)
        width_diff = sub_df["width"].diff().fillna(0).abs()
        similar_width = (width_diff / sub_df["width"]) <= 0.01

        previous_has_sentence_end = previous_text.str.rstrip()\
            .str[-1].isin([".", "?", "!"])  
        
        current_starts_capital = current_text.str.lstrip().str[0].str.isupper()
        continuation = (
            similar_width
            & ~previous_has_sentence_end
            & ~current_starts_capital.fillna(False)
        )
        sub_df.loc[continuation, "is_article_start"] = False

        # Apply the capitalization-based filtering to the article splits
        df.loc[sub_df.index, "is_article_start"] = sub_df["is_article_start"]
        return df
    

    def segment_paragraphs(
            self,
            df,
            block_filter: Sequence[str] | None = None
        ) -> pd.DataFrame:
        """
        Segment the text lines into paragraphs based on horizontal indentations, 
        compared to the text block's border positions.
        Uses relative difference, and z-scores of those differences to determine 
        paragraph indentations.

        Args:
            df (pd.DataFrame): DataFrame containing text lines with article identifiers.
            block_filter: Optional list of block IDs to restrict processing to.

        Returns:
            pd.DataFrame: A DataFrame containing the text lines with paragraph identifiers.
        """
 
        block_filter = self.block_filter if block_filter is None else list(block_filter)
        self.text_blocks_df = self.get_text_blocks_df(block_filter=block_filter)


        group_columns = [self.block_id_col]

        if self.article_segmentation:
            group_columns.append(self.article_id_output_col)

        grouped = df.groupby(group_columns, sort=False)
        group_order = grouped.ngroup()


        # Recalculate block_hpos and block_width per block, by using the min hpos 
        # and max hpos within each block. (sometimes the block boundaries were 
        # not based on the individual line positions)
        block_hpos = grouped["hpos"].transform("min")
        block_width = grouped["width"].transform("max")

        hpos_pct_diff = (df["hpos"] - block_hpos) / block_width

        outliers = hpos_pct_diff.abs() > 0.15

        # Exclude outliers from each group's z-score calculation.
        hpos_z = self._group_zscore(
            df, hpos_pct_diff.where(~outliers), group_columns
        )

        # df["hpos_pct_diff"] = hpos_pct_diff
        # df["hpos_z"] = hpos_z

        paragraph_indent = (
            (hpos_z >= self.paragraph_split_threshold_z)
            & (hpos_pct_diff >= self.paragraph_split_threshold_r)
        )
        df["is_paragraph_start"] = paragraph_indent | outliers

        # Lines with 'hyppart2', (second part of a hyphenated word) cannot be the 
        # start of a new paragraph.
        if 'hyppart2' in df.columns:
            df.loc[df['hyppart2'].notna(), "is_paragraph_start"] = False


        if self.paragraph_single_line_split:
            single_line = grouped[group_columns[0]].transform("size").eq(1)
            df.loc[single_line, "is_paragraph_start"] = True

        df = df.assign(_group_order=group_order)
        df = df.sort_values("_group_order", kind="stable").drop(columns="_group_order")
        df = df.reset_index(drop=True)

        text_blocks = self.text_blocks_df.set_index("id")
        same_column = df[self.block_id_col].map(text_blocks["same_column"])
        first_line = ~df[self.block_id_col].duplicated(keep="first")
        df.loc[
            same_column & ~first_line & df["is_article_start"],
            "is_paragraph_start",
        ] = True

        df[self.paragraph_id_output_col] = (
            "P" + df["is_paragraph_start"].cumsum().astype(str).str.zfill(3)
        )

        self.paragraph_df = df.copy()
        return df

    def load_alto_xml(
            self, 
            alto_xml_path: str | Path | etree._Element,
            use_cache: bool = False,
        ) -> etree._Element:
        """
        Load an ALTO XML file from a given path or URL and return the parsed XML tree.

        Args:
            alto_xml_path (str|Path|etree._Element): The path to the ALTO XML file, 
                a URL, or an already parsed XML element (which will be returned as is).

        Returns:
            etree._Element: The parsed ALTO XML tree.
        """
        use_cache = use_cache or self.use_alto_cache

        ID_PATTERN = r'[\w\d]{3,}:\d{9}:mpeg21:p\d{2,}'

        if isinstance(alto_xml_path, etree._Element):
            return alto_xml_path

        if Path(alto_xml_path).is_file():
            return etree.fromstring(Path(alto_xml_path).read_bytes())

        if not isinstance(alto_xml_path, str):
            raise ValueError(f"Invalid ALTO XML path or URL: {alto_xml_path}")

        if re.match(ID_PATTERN, alto_xml_path):
            alto_url = f"https://resolver.kb.nl/resolve?urn={alto_xml_path}:alto"
        elif validators.url(alto_xml_path):
            alto_url = alto_xml_path
        else:
            raise ValueError(f"Invalid ALTO XML path or URL: {alto_xml_path}")

        if use_cache and self.alto_cache_dir:
            alto_id = alto_url.split("urn=")[-1].replace(":", "_") + ".xml"
            alto_cache_path = Path(self.alto_cache_dir) / alto_id
            if alto_cache_path.is_file():
                return etree.fromstring(alto_cache_path.read_bytes())

        response = requests.get(alto_url)
        response.raise_for_status()
        alto_xml_tree = etree.fromstring(response.content)

        if use_cache and self.alto_cache_dir and alto_cache_path:
            alto_cache_path.write_bytes(response.content)

        return alto_xml_tree


    def get_text_blocks_df(
            self,
            block_filter: Sequence[str] | None = None,
        ) -> pd.DataFrame:
        """
        Get a DataFrame containing the attributes of all TextBlock elements in the ALTO XML tree.

        Args:
            block_filter (Sequence[str] | None): List of block IDs to include. If empty, include all blocks.

        Returns:
            pd.DataFrame: A DataFrame containing the attributes of the TextBlock elements.

        """
        text_bs = self.alto_xml_tree.findall(".//TextBlock", namespaces=self.nsmap)
        text_bs_df = self.elms_to_df(text_bs)
        text_bs_df = self.check_logical_flow_between_blocks(text_bs_df)

        if block_filter:
            text_bs_df = text_bs_df[text_bs_df['id'].isin(block_filter)]

        return text_bs_df

    def get_text_styles_df(self) -> pd.DataFrame:
        """
        Get a DataFrame containing the attributes of all TextStyle elements in the ALTO XML tree.

        Returns:
            pd.DataFrame: A DataFrame containing the attributes of the TextStyle elements.
        """
        text_styles = self.alto_xml_tree.findall(".//Styles/TextStyle", namespaces=self.nsmap)
        df = self.elms_to_df(text_styles)
        return df

    def get_paragraph_styles_df(self) -> pd.DataFrame:
        """
        Get a DataFrame containing the attributes of all ParagraphStyle elements in the ALTO XML tree.

        Returns:
            pd.DataFrame: A DataFrame containing the attributes of the ParagraphStyle elements.
        """
        par_styles = self.alto_xml_tree.findall(".//Styles/ParagraphStyle", namespaces=self.nsmap)
        df = self.elms_to_df(par_styles)
        return df

    def get_text_lines_df(
            self,
            block_filter: Sequence[str] | None = None,
            add_strings: bool = True,
            keep_styles: bool = True,
        ) -> pd.DataFrame:
        """
        Get a DataFrame containing the attributes of all TextLine elements in the ALTO XML tree.

        Args:
            block_filter (Sequence[str] | None): List of block IDs to include. If empty,
                include all blocks.
            add_strings (bool): Whether to include the text from String elements within each TextLine.
            keep_styles (bool): Whether to keep style information from the ALTO XML.

        Returns:
            pd.DataFrame: A DataFrame containing the attributes of the TextLine elements, 
                with optional string and style information.
        """
        blocks = self.alto_xml_tree.findall(".//TextBlock", namespaces=self.nsmap)

        if block_filter:
            allowed_ids = set(block_filter)
            blocks = [block for block in blocks if block.get("ID") in allowed_ids]

        line_elements = []
        line_block_ids = []
        string_records = []

        for block in blocks:
            block_id = block.get("ID")
            for line in block.findall(".//TextLine", namespaces=self.nsmap):
                line_elements.append(line)
                line_block_ids.append(block_id)

                if add_strings:
                    for string in line.findall(".//String", namespaces=self.nsmap):
                        record = dict(string.attrib)
                        record["LINE_ID"] = line.get("ID")
                        string_records.append(record)

        if not line_elements:
            return pd.DataFrame()

        # Construct one line DataFrame for the page.
        df = self.elms_to_df(line_elements, keep_styles=keep_styles)
        df["block_id"] = line_block_ids

        block_meta = self.elms_to_df(blocks, keep_styles=keep_styles)
        block_meta = block_meta.set_index("id")

        df["block_hpos"] = df["block_id"].map(block_meta["hpos"]).astype(int)
        df["block_width"] = df["block_id"].map(block_meta["width"]).astype(int)

        if add_strings:
            if not string_records:
                df["text"] = pd.NA
                df["hyphenated"] = False
                return df

            strings_df = self.elms_to_df(string_records, keep_styles=keep_styles)

            if keep_styles:
                line_styles = df.set_index("id")
                for col in ("font_style", "par_style"):
                    if col not in strings_df.columns:
                        strings_df[col] = pd.NA
                    if col in line_styles.columns:
                        strings_df[col] = strings_df[col].fillna(
                            strings_df["line_id"].map(line_styles[col])
                        )

            agg = {"text": ("content", lambda values: " ".join(values.dropna()))}

            if "subs_type" in strings_df.columns:
                subs_df = (
                    strings_df.pivot(columns="subs_type", values="content")
                    .filter(like="Hyp")
                    .rename(columns=str.lower)
                )
                strings_df = strings_df.merge(
                    subs_df, left_index=True, right_index=True
                )

                strings_df["hyphenated"] = False
                if "hyppart1" in strings_df.columns:
                    strings_df["hyphenated"] = strings_df["hyppart1"].notna()

                for col in subs_df.columns:
                    agg[col] = (col, "first")

                agg["hyphenated"] = ("hyphenated", "any")
                agg["hypcontent"] = ("subs_content", "first")

            if keep_styles:
                def most_common(values):
                    counts = values.value_counts(dropna=False)
                    return counts.index[0] if not counts.empty else pd.NA

                for col in ("par_style", "font_style"):
                    if col in strings_df.columns:
                        agg[col] = (col, most_common)

            self._strings_df = strings_df.copy()
            line_text = (
                strings_df.groupby("line_id", sort=False)
                .agg(**agg)
                .reset_index()
            )

            if "hyphenated" not in line_text.columns:
                line_text["hyphenated"] = False

            df = df.merge(line_text, how="left", left_on="id", right_on="line_id")

        return df

    @staticmethod
    def elms_to_df(
            elements: Sequence[etree._Element | Mapping[str, Any] | etree._Attrib],
            drop_empty_columns: bool = True,
            font_style_start: str = "T",
            keep_styles: bool = True,
        ) -> pd.DataFrame:
        """
        Create a dataframe with the attributes of all the xml elements in the input list.

        Args:
            elements (list): A list of XML elements (lxml.etree._Element) or dictionaries.
            drop_empty_columns (bool): Whether to drop columns that are completely empty.
            font_style_start (str): The starting string to identify font styles, 
                when splitting stylerefs. The other style part will be identified 
                as paragraph style.

        Returns:
            pd.DataFrame: A DataFrame containing the attributes of the XML elements, with
                column names converted to lowercase. If the input list is empty,
                an empty DataFrame is returned.
        """

        if not elements:
            return pd.DataFrame()
        
        records: list[Mapping[str, Any]] = []
        for el in elements:
            if isinstance(el, etree._Element):
                records.append(el.attrib)
            elif isinstance(el, (Mapping, etree._Attrib)):
                records.append(el)
            else:
                raise TypeError(
                    f"Unsupported element type {type(el).__name__} in elements. "
                    "Expected etree._Element, etree._Attrib, or a Mapping[str, Any] (e.g. dict)."
                )
        
        df = pd.DataFrame(records)
        df.columns = [i.lower() for i in df.columns]

        if 'stylerefs' in df.columns and keep_styles:
            # Split into font_style and par_style columns, if present
            def split_stylerefs(styleref: str) -> dict[str, str]:

                font_style = pd.NA
                par_style = pd.NA

                if styleref and isinstance(styleref, str):
                    styles = styleref.split()
                    for style in styles:
                        if style.startswith(font_style_start):
                            font_style = style
                        else:
                            par_style = style

                return {"font_style": font_style, "par_style": par_style}
            
            styles = df['stylerefs'].apply(split_stylerefs)

            df[['font_style', 'par_style']] = pd.DataFrame(styles.tolist(), index=df.index)


        if drop_empty_columns:
            df = df.dropna(axis=1, how='all')

        for col in df.columns:
            if df[col].dropna().str.match(r"^\d+(\.\d+)?$").all():
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


    @staticmethod
    def check_logical_flow_between_blocks(
            df: pd.DataFrame,
            h_overlap_threshold: float = 0.8,
            clean_up: bool = True,
        ) -> pd.DataFrame:
        """Check logical flow between consecutive blocks using horizontal overlap.
        
        It checks if:
        - A consecutive block is in the same column (based on horizontal overlap)
        - A consecutive block starts to the right of the previous block.
        
        If any of those conditions are met, there is a logical flow between the 
        consecutive blocks. Meaning the text of the previous block probably logically 
        continues into the next block.
        
        Args:
            df (pd.DataFrame): DataFrame containing the blocks to check for logical flow.
            h_overlap_threshold (float, optional): Threshold for horizontal overlap to 
                consider blocks in the same column. Defaults to 0.8.
            clean_up (bool, optional): Whether to remove intermediate columns used 
                for calculation. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with additional columns indicating logical flow between blocks.
        
        """
        input_columns = df.columns.tolist()
        df = df.copy()

        left = df['hpos'].to_numpy(dtype=float)
        right = (df['hpos'] + df['width']).to_numpy(dtype=float)

        previous_left = np.roll(left, 1)
        previous_right = np.roll(right, 1)
        if len(df):
            # Match the original behavior: the first block is compared with itself.
            previous_left[0] = left[0]
            previous_right[0] = right[0]

        intersection = np.maximum(
            0.0,
            np.minimum(right, previous_right) - np.maximum(left, previous_left),
        )

        current_length = np.abs(right - left)
        previous_length = np.abs(previous_right - previous_left)

        current_overlap = np.divide(
            intersection,
            current_length,
            out=np.zeros_like(intersection),
            where=current_length != 0,
        )
        previous_overlap = np.divide(
            intersection,
            previous_length,
            out=np.zeros_like(intersection),
            where=previous_length != 0,
        )

        df['pct_overlap'] = np.maximum(current_overlap, previous_overlap)
        df['same_column'] = df['pct_overlap'] >= h_overlap_threshold
        df['starts_right'] = df['hpos'] >= df['hpos'].shift(1)
        df['logical_flow'] = df['same_column'] | df['starts_right']

        if clean_up:
            df = df[input_columns + ['same_column', 'starts_right', 'logical_flow']]

        return df


    def get_lines_in_block(
            self,
            block: etree._Element,
            add_strings: bool = True,
            keep_styles: bool = False,
        ) -> pd.DataFrame:
        """Get all TextLine elements in a given TextBlock element and return them as a DataFrame.
        
        If add_strings is True, also extract the String elements from each TextLine and add a
        'text' column to the DataFrame containing the concatenated content of the String elements.

        Args:
            block (etree._Element): The TextBlock element from which to extract TextLine elements.
            add_strings (bool): If True, also extract String elements from each 
                TextLine and add a 'text' column. Defaults to True.
            keep_styles (bool): If True, retain style information from the XML elements. 
                Defaults to False.

        """
        text_lines = block.findall(".//TextLine", namespaces=self.nsmap)
        df = self.elms_to_df(text_lines, keep_styles=keep_styles)

        block_meta = self.elms_to_df([block], keep_styles=keep_styles).iloc[0]
        df['block_hpos'] = int(block_meta['hpos'])
        df['block_width'] = int(block_meta['width'])
        df['block_id'] = block_meta['id']

        if add_strings:
            strings = []
            for line in text_lines:
                line_strings = [dict(s.attrib) for s in line.findall(".//String", namespaces=self.nsmap)]
                for s in line_strings:
                    s["LINE_ID"] = line.attrib["ID"]
                strings.extend(line_strings)

            if keep_styles:
                lines_styles = pd.DataFrame(index=df['id'].values)
                if 'stylerefs' in df.columns:
                    lines_styles = df.filter(['id', 'font_style', 'par_style'])\
                        .set_index('id')
                    
                for col in ["font_style", "par_style"]:
                    if col not in lines_styles.columns:
                        lines_styles[col] = pd.NA

                    lines_styles[col] = lines_styles[col]\
                        .fillna(block_meta.get(col, pd.NA))
                
            strings_df = self.elms_to_df(strings, keep_styles=keep_styles)

            if "line_id" not in strings_df.columns:
                print("Warning: No line_id column found in strings_df. Check the ALTO XML structure.")
                print(strings_df.head())

            # Define aggregation functions for merging strings into lines
            agg = {
                "text" : ("content", " ".join),
            }

            if 'subs_type' in strings_df.columns:
                subs_df = strings_df.pivot(columns='subs_type', values='content')\
                            .filter(like='Hyp')\
                            .rename(columns=str.lower)
                
                for col in subs_df.columns:
                    agg[col] = (col, "first")
                
                strings_df = strings_df.merge(
                    subs_df,
                    left_index=True,
                    right_index=True,
                )

                strings_df['hyphenated'] = False
                if 'hyppart1' in strings_df.columns:
                    strings_df['hyphenated'] = strings_df['hyppart1'].notna()

                agg['hyphenated'] = ('hyphenated', "any")
                agg['hypcontent'] = ('subs_content', "first")

            if keep_styles:
                # Also add stylerefs and font_style, if present; keep most common value
                def get_style(x: pd.Series) -> str | None:
                    if x.empty:
                        return pd.NA
                    style_counts = x.value_counts(dropna=False)
                    return style_counts.index[0]
                
                    # unique_styles = x.nunique()
                    # if len(unique_styles) == 1:
                    #     return unique_styles[0]
                    # else:
                    #     return pd.NA
                    
                for col in ["par_style", "font_style"]:
                    if col not in strings_df.columns:
                        # strings_df[col] = block_meta.get(col, pd.NA)
                        strings_df[col] = strings_df['line_id'].map(lines_styles[col])
                    else:
                        # strings_df[col] = strings_df[col].fillna(block_meta.get(col, pd.NA))
                        strings_df[col] = strings_df[col].fillna(
                            strings_df['line_id'].map(lines_styles[col])
                        )

                    agg[col] = (col, get_style)

            self._strings_df = strings_df.copy()

            line_text = strings_df.groupby("line_id", sort=False)\
                .agg(**agg)\
                .reset_index()
            
            if 'hyphenated' not in line_text.columns:
                line_text['hyphenated'] = False
            
            df = df.merge(line_text, how='left', left_on='id', right_on='line_id')

        return df


    def find_article_starts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find article splits in a block of text lines based on vertical position differences.
        """
        df['line_top'] = df['vpos'] + df['height']
        df['vpos_diff'] = df['vpos'] - df['line_top'].shift(1)
        df['vpos_diff_z'] = zscore(df['vpos_diff'], nan_policy='omit')

        min_difference = df.height.median() // 3
        df['is_article_start'] = (df['vpos_diff_z'] >= self.article_split_threshold) & (
            df['vpos_diff'] >= min_difference
        )

        if self.article_single_line_split and len(df) == 1:
            df['is_article_start'] = True

        if self.clean_up:
            df.drop(columns=['line_top', 'vpos_diff', 'vpos_diff_z'], inplace=True)

        return df


    def find_paragraph_starts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find paragraph start lines in a block of text lines based on horizontal position differences,
        using z-score normalization to identify outliers that indicate paragraph breaks. 
        
        Paragraph starts should also have a minimal horizontal position difference threshold,
        to avoid false positives in case the horizontal position difference is very small 
        but the z-score is high due to low variance.
        
        """
        df = df.copy()

        # Calculate percentage difference between the horizontal position of the line and the block
        df['hpos_pct_diff'] = (df['hpos'] - df['block_hpos']) / df['block_width']
        # Define outliers, to be excluded from the z-score calculation, as they can skew the results
        # usually these are headers or other lines that are not part of the main text flow
        outliers = df['hpos_pct_diff'].abs() > 0.15

        # z-score calculation
        df.loc[~outliers, 'hpos_z'] = zscore(df.loc[~outliers, 'hpos_pct_diff'], nan_policy='omit')

        paragraph_indent = (df['hpos_z'] >= self.paragraph_split_threshold) \
            & (df['hpos_pct_diff'] >= self.paragraph_minimal_horizontal_position_difference)
        
        df['is_paragraph_start'] = paragraph_indent | outliers

        if self.paragraph_single_line_split and len(df) == 1:
            df['is_paragraph_start'] = True

        if self.clean_up:
            df.drop(columns=['hpos_z'], inplace=True)

        return df


    def create_new_text_blocks_df(
            self,
            df: pd.DataFrame|None = None,
        ) -> pd.DataFrame:
        """
        Create a new DataFrame containing the coordinates of the new text blocks
        based on the article and paragraph segmentation.

        Args:
            df (pd.DataFrame): DataFrame containing the segmented text lines with 
                article and paragraph identifiers.

        Returns:
            pd.DataFrame: A DataFrame containing the coordinates of the new text blocks.
        """

        if isinstance(df, type(None)):
            df = self.df

        def get_block_coords(df : pd.DataFrame) -> pd.Series:
            last_line = df.iloc[-1]
            # print(df.index, df.name)

            x1_min = df['hpos'].min()
            x2_pos = df['hpos'] + df['width']
            width = (x2_pos - x1_min).max()

            height = last_line['vpos'] + last_line['height'] - df['vpos'].min()

            block_coords = {
                # 'id' : last_line['article_id'],
                # 'original_block_id' : last_line['block_id'],
                'hpos' : df['hpos'].min(),
                'vpos' : df['vpos'].min(),
                'width' : width,
                'height' :height,
            }
            if 'page_id' in df.columns:
                block_coords['page_id'] = df['page_id'].iloc[0]
            return pd.Series(block_coords)

        blocks_df = df.groupby(
                [self.block_article_id_col, self.block_id_col, self.article_id_output_col],
                as_index=False
            ).apply(get_block_coords)

        self.new_text_blocks_df = blocks_df

        return blocks_df


    def merge_paragraphs(
            self,
            df: pd.DataFrame,
            group_columns: Sequence[str] | None = None,
            output_column: str | None = 'text',
        ) -> pd.DataFrame:
        """
        Merge text lines into paragraphs based on the specified paragraph ID column.
        Also handles hyphenated words by removing the hyphen and concatenating the words.

        Args:
            df (pd.DataFrame): DataFrame containing text lines with the 
                paragraph ID column and 'text' columns.
            group_columns (Sequence[str], optional): The list of columns to group by.
                If None, defaults to [self.article_id_output_col, self.paragraph_id_output_col].
            output_column (str, optional): The name of the column to store the merged text. 
                If None, defaults to 'text'.

        Returns:
            pd.DataFrame: A DataFrame containing merged paragraphs with the paragraph ID column 
                (as specified by group_columns), and 'text' columns.
        """
        group_columns = group_columns or [self.article_id_output_col, self.paragraph_id_output_col]
        output_column = output_column or 'text'

        hyp_cols = list(df.filter(regex="hyp").columns)
        filter_columns = list(group_columns) + hyp_cols + ['text', "line_id", self.block_article_id_col]
        df = df.filter(items=filter_columns).copy()

        if 'hyphenated' in df.columns:
            # In lines with a values in 'hyppart1', replace that value in the text (at the end of the line),  
            # with the 'hypcontent' value (which is the full word). In lines with a value in 'hyppart2', 
            # remove that value from the start of the text string, and strip any leading whitespace.

            # 1. Replace values in 'hyppart1' with the full word from 'hypcontent'.
            if 'hyppart1' in df.columns:
                mask = df['hyppart1'].notna()
                df.loc[mask, 'text'] = df.loc[mask].apply(
                    lambda row: row['text'].rsplit(row['hyppart1'], 1)[0] + row['hypcontent'], 
                    axis=1
                )
                
            # 2. Remove values in 'hyppart2' from the start of the text and strip leading whitespace.
            if 'hyppart2' in df.columns:
                mask = df['hyppart2'].notna()
                df.loc[mask, 'text'] = df.loc[mask].apply(
                    lambda row: row['text'][len(row['hyppart2']):].lstrip(), 
                    axis=1
                )
        
        paragraphs_df = df.groupby(group_columns, sort=False).agg(
            {
                output_column: lambda x: ' '.join(x.dropna()),
                self.block_article_id_col: 'unique',
                "line_id": list
            }
        ).reset_index()

        return paragraphs_df


    @staticmethod
    def merge_paragraphs_over_page_split(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge paragraphs spanning over two pages. If the first paragraph on on the
        next page doesn't start with a paragraph break (i.e. paragraph_id ends with 0),
        we assume that this paragraph is a continuation of the previous paragraph, from
        the previous page.
        In that case, we merge the text of the two paragraphs and drop the first paragraph 
        on the next page.

        Args:
            df (pd.DataFrame): DataFrame containing the paragraphs with 
                columns 'page_idx', 'paragraph_id', and 'text'.
        
        Returns:
            pd.DataFrame: DataFrame with merged paragraphs and dropped P0 paragraphs.

        """
        df = df.reset_index(drop=True)

        for idx, row in df.iterrows():
            p_nr = int(str(row['paragraph_id'])[1:])

            if p_nr == 0 and idx != 0:
                df.loc[idx - 1, 'text'] += ' ' + df.loc[idx, 'text']
                df = df.drop(idx).reset_index(drop=True)

        df = df.drop(columns=['article_id', 'paragraph_id'])

        return df

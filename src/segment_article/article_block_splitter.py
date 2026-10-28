import cv2
import numpy as np
import pandas as pd
from PIL import Image
import shapely
from shapely import ops

class ArticleBlockSplitter:
    """Split image regions into smaller text blocks using horizontal rules.

    Detector settings are stored on the instance so the same configuration can
    be applied to multiple pages.
    """

    @staticmethod
    def convert_image_to_array(img: Image.Image|np.ndarray) -> np.ndarray:
        """Convert an PIL image to a numpy array. If the input is already a numpy
        array, it is returned as is.

        Args:
            img (PIL.Image or np.ndarray): The input image.

        Returns:
            np.ndarray: The converted image as a numpy array.
        """

        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            return np.ascontiguousarray(np.array(img.convert("L"), dtype=np.uint8))
        raise ValueError("Unsupported image type")

    def __init__(
        self,
        min_line_width_ratio: float = 0.1,
        max_line_gap: int = 10,
        hough_threshold: int = 40,
        center_tolerance: float = 0.1,
    ) -> None:
        self.min_line_width_ratio = min_line_width_ratio
        self.max_line_gap = max_line_gap
        self.hough_threshold = hough_threshold
        self.center_tolerance = center_tolerance

    def split_blocks(
        self,
        lines_df: pd.DataFrame,
        blocks_df: pd.DataFrame,
        img: Image.Image | np.ndarray,
    ) -> dict[str, pd.DataFrame]:
        """Split multiple blocks and update their associated line IDs.

        The input line dataframe is copied. The returned dictionary contains
        the updated ``lines_df`` and the resulting ``blocks_df``.
        """
        updated_lines_df = lines_df.copy()
        all_split_blocks = []
        image_array = self.convert_image_to_array(img)

        for _, block_row in blocks_df.iterrows():
            try:
                result = self.split_block(
                    df=updated_lines_df,
                    row=block_row,
                    img=image_array,
                )
            except Exception as error:
                print(f"Error splitting block '{block_row['id']}': \n{error}")
                print(block_row)
                raise

            updated_lines_df = result["lines_df"]
            all_split_blocks.append(result["blocks_df"])

        split_blocks_df = pd.concat(all_split_blocks, ignore_index=True)
        return {"lines_df": updated_lines_df, "blocks_df": split_blocks_df}

    def split_block(
        self,
        df: pd.DataFrame,
        row: pd.Series | dict | pd.DataFrame,
        img: Image.Image | np.ndarray,
    ) -> dict[str, pd.DataFrame]:
        """Split one block and assign its lines to the resulting block IDs.

        A multirow dataframe is treated as a request to split multiple blocks.
        This method mutates ``df`` and returns it in the ``lines_df`` field.
        """
        image_array = self.convert_image_to_array(img)

        if isinstance(row, dict):
            row = pd.Series(row)

        if isinstance(row, pd.DataFrame):
            if len(row) == 1:
                row = row.iloc[0]
            else:
                return self.split_blocks(lines_df=df, blocks_df=row, img=image_array)

        original_block_id = row["id"]
        line_mask = df["block_id"] == original_block_id

        # A single-line block cannot contain an article split.
        if line_mask.sum() <= 1:
            df.loc[line_mask, "original_block_id"] = original_block_id
            row["original_id"] = original_block_id
            return {"lines_df": df, "blocks_df": row.to_frame().T}

        lines = self.detect_horizontal_lines_in_block(
            img=image_array,
            block_row=row,
            min_line_width_ratio=self.min_line_width_ratio,
            max_line_gap=self.max_line_gap,
            hough_threshold=self.hough_threshold,
            return_image=False,
        )

        if not lines:
            df.loc[line_mask, "original_block_id"] = original_block_id
            row["original_id"] = original_block_id
            return {"lines_df": df, "blocks_df": row.to_frame().T}

        line_polygons = self.to_polygon(lines)
        grouped_lines = self.group_lines(line_polygons)
        final_line_polygons = self.filter_non_centered_lines(
            grouped_lines,
            block_row=row,
            tolerance=self.center_tolerance,
        )
        alto_lines = self.polygon_to_alto_coords(final_line_polygons)
        split_textblocks_df = self.split_block_by_lines(row, alto_lines)

        block_lines_new = self.assign_lines_to_new_blocks(
            df.loc[line_mask],
            split_textblocks_df,
        )
        df.loc[line_mask, "original_block_id"] = original_block_id
        df.loc[line_mask, "block_id"] = block_lines_new["block_id"].values

        return {"lines_df": df, "blocks_df": split_textblocks_df}


    @staticmethod
    def detect_horizontal_lines_in_block(
            img,
            block_row=None,
            min_line_width_ratio=0.1,
            max_line_gap=10,
            hough_threshold=40,
            return_image=False,
            print_debug=False,
        ):
        """
        Detect horizontal lines in a given block of an image using OpenCV's Hough Line Transform.

        Args:
            img (PIL.Image or np.ndarray): The input image in which to detect lines.
            block_row (pd.Series or dict): A series or dictionary containing the block's
                'hpos', 'vpos', 'width', and 'height' to define the region of interest (ROI) for line detection.
            min_line_width_ratio (float): The minimum length of a line to be considered, as a ratio of the block's width.
            max_line_gap (int): The maximum allowed gap between line segments to treat them as a single line.
            hough_threshold (int): The threshold parameter for the Hough Line Transform.
            return_image (bool): If True, return the intermediate images (gray, thresh, morph) along with the detected lines.
            print_debug (bool): If True, print debug information about the detected lines.

        Returns:
            list of tuples: A list of detected horizontal lines, each represented as a tuple (x1, y1, x2, y2).
            If return_image is True, returns a dictionary with the detected lines and intermediate images.
        """

        # split_img
        CROPPED = False

        if isinstance(img, Image.Image):
            img = img.convert("L")
            # Convert to NumPy array for OpenCV processing
            img = np.array(img)

        split_img = img

        if isinstance(block_row, pd.Series|dict):
            block = block_row

            # Crop out block from the original image
            split_img = img[block['vpos']:block['vpos'] + block['height'], block['hpos']:block['hpos'] + block['width']]
            # Convert to grayscale for OpenCV processing
            if len(split_img.shape) == 3:
                split_img = cv2.cvtColor(split_img, cv2.COLOR_BGR2GRAY)

            CROPPED = True

        # Convert the PIL image to a NumPy array for OpenCV processing
        # roi = np.array(split_img)
        gray = split_img.copy()

        # 2. Binariseren (Otsu thresholding om papierruis te negeren)
        # Krantenpapier heeft vaak vlekken; Otsu of Adaptive Threshold helpt enorm.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Optioneel: Morphological kernel om alleen horizontale structuren te versterken
        # Dit filtert kleine spikkels en verticale ruis eruit.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 4. Zoek naar lijnen met de Probabilistic Hough Line Transform
        min_line_length = int(gray.shape[1] * min_line_width_ratio) # Minimale lengte van de lijn
        # Maximale onderbreking in de lijn (door vervaagde inkt)

        lines = cv2.HoughLinesP(
            morph,
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        # 5. Controleer of er een (vrijwel) horizontale lijn tussen zit
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line#[0]
                # Bereken de hoek; we willen alleen (bijna) horizontale lijnen
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if angle < 5 or angle > 175: # Binnen 5 graden van horizontaal
                    # print(f"Found horizontal line")
                    if CROPPED:
                        # Convert the coordinates back to the original image coordinates
                        x1 = x1 + block['hpos']
                        x2 = x2 + block['hpos']
                        y1 = y1 + block['vpos']
                        y2 = y2 + block['vpos']

                    filtered_lines.append((x1, y1, x2, y2))

        if print_debug:
            print(f"Detected {len(filtered_lines)} horizontal lines in block {block['id']}")

        if return_image:
            return {"lines": filtered_lines, "gray": gray, "thresh": thresh, "morph": morph}
        return filtered_lines


    @staticmethod
    def to_polygon(line_coords):
        """
        Convert a list of line coordinates to shapely polygons.

        Args:
            line_coords (list of tuples): A list of tuples representing line coordinates.
                Each tuple should contain (x1, y1, x2, y2) coordinates of a line.

        Returns:
            list of shapely.geometry.Polygon: A list of shapely polygons representing the lines.
        """
        # Convert each line to a shapely polygon
        line_polygons = []
        for line in line_coords:
            x1, y1, x2, y2 = line
            # Create a polygon with a small height (e.g., 5 pixels) to represent the line
            line_polygon = shapely.geometry.box(x1, y1 - 2, x2, y2 + 2)
            line_polygons.append(line_polygon)
        return line_polygons


    @staticmethod
    def group_lines(line_polygons):
        """
        Group overlapping line polygons into merged polygons.

        Args:
            line_polygons (list of shapely.geometry.Polygon): A list of line polygons.

        Returns:
            list of shapely.geometry.Polygon: A list of merged polygons representing
                grouped lines.
        """

        # Copy line_polygons, so we don't modify the original list
        line_polygons = line_polygons.copy()

        merged = []
        while line_polygons:
            current = line_polygons.pop(0)
            overlaps = [current]
            for other in line_polygons[:]:
                if current.intersects(other):
                    overlaps.append(other)
                    line_polygons.remove(other)
            # Merge all overlapping lines into a single polygon
            merged_polygon = ops.unary_union(overlaps)
            merged.append(merged_polygon)
        return merged



    @staticmethod
    def split_block_by_lines(block_row, lines):
        """
        Split a block into multiple blocks based on the lines found within it.

        Args:
            block_row (pd.Series): A row from the DataFrame representing the original block.
            lines (list of dict): A list of dictionaries representing the lines found within the block.
                Each dictionary should contain 'hpos', 'vpos', 'width', and 'height' keys.

        Returns:
            pd.DataFrame: A DataFrame containing the new blocks created by splitting the original block.
        """
        new_blocks = []
        # Sort lines by vertical position
        lines_sorted = sorted(lines, key=lambda x: x['vpos'])

        # Initialize previous line's bottom position
        prev_bottom = block_row['vpos']

        base_block = {
            "original_id": block_row['id'],
            'hpos': block_row['hpos'],
            'width': block_row['width'],
        }

        for idx, line in enumerate(lines_sorted):
            nr = idx + 1
            new_id = f"{block_row['id']}_{nr}"
            # Create a new block from the previous bottom to the current line's top
            new_block = base_block.copy()
            new_block.update({
                'id' : new_id,
                'vpos': prev_bottom,
                'height': line['vpos'] - prev_bottom
            })

            new_blocks.append(new_block)

            # Update previous bottom to current line's bottom
            prev_bottom = line['vpos'] + line['height']

        # Add the last block from the last line's bottom to the original block's bottom
        if prev_bottom < block_row['vpos'] + block_row['height']:
            new_block = base_block.copy()
            new_block.update({
                'id' : f"{block_row['id']}_{len(lines_sorted) + 1}",
                'vpos': prev_bottom,
                'height': (block_row['vpos'] + block_row['height']) - prev_bottom
            })

            new_blocks.append(new_block)

        new_blocks_df = pd.DataFrame(new_blocks)
        # Remove rows with height <= 0, as they are not valid blocks
        new_blocks_df = new_blocks_df[new_blocks_df['height'] > 0].reset_index(drop=True)

        # Add columns of the original block_row
        for col in block_row.index:
            if col not in new_blocks_df.columns:
                value = block_row[col]
                # Value can be a list (like) with values, so make sure value is
                # treated as a single value for all new blocks.
                new_blocks_df[col] = [value for _ in range(len(new_blocks_df))]


        return new_blocks_df

    @staticmethod
    def filter_non_centered_lines(lines, block_row, tolerance=0.1):
        """
        Filter out lines that are not centered within the block's width. The centroid
        of a line should be within a certain tolerance of the block's center.

        Args:
            lines (list of shapely.geometry.Polygon): A list of line polygons.
            block_row (pd.Series): A row from the DataFrame representing the original block.
            tolerance (float): The tolerance for determining if a line is centered.
                A value of 0.1 means the line's centroid must be within 10% of the block's width from the center.

        Returns:
            list of shapely.geometry.Polygon: A filtered list of line polygons that are centered within the block.
        """
        block_center_x = block_row['hpos'] + block_row['width'] / 2
        filtered_lines = []
        for line in lines:
            line_centroid_x = line.centroid.x
            if abs(line_centroid_x - block_center_x) <= tolerance * block_row['width']:
                filtered_lines.append(line)
        return filtered_lines


    @staticmethod
    def polygon_to_alto_coords(
            poly : shapely.geometry.Polygon|list
        ) -> dict|list:
        """
        Convert a shapely polygon to ALTO coordinates, with 'hpos', 'vpos', 'width',
        and 'height'.

        Args:
            poly (shapely.geometry.Polygon or list): A shapely polygon or a list of polygons.

        Returns:
            dict or list: A dictionary with 'hpos', 'vpos', 'width', and 'height' if a
                single polygon is provided, or a list of such dictionaries if a list
                of polygons is provided.
        """

        if isinstance(poly, list):
            return [ArticleBlockSplitter.polygon_to_alto_coords(p) for p in poly]

        minx, miny, maxx, maxy = poly.bounds

        alto_coords = {
            'hpos': int(minx),
            'vpos': int(miny),
            'width': int(maxx - minx),
            'height': int(maxy - miny)
        }
        return alto_coords


    @staticmethod
    def assign_lines_to_new_blocks(lines_df, new_blocks_df):
        """
        Assign TextLines to new blocks based on their vertical position.

        Args:
            lines_df (pd.DataFrame): DataFrame containing the original TextLines with 'vpos' and 'height'.
            new_blocks_df (pd.DataFrame): DataFrame containing the new blocks with 'vpos' and 'height'.

        Returns:
            pd.DataFrame: The original lines_df with an additional column 'new_block_id' indicating the assigned block.
        """
        lines_df = lines_df.copy()
        lines_df['original_block_id'] = lines_df['block_id']
        lines_df['block_id'] = None

        for _, block in new_blocks_df.iterrows():
            block_top = block['vpos']
            block_bottom = block['vpos'] + block['height']

            # Assign lines that fall within the block's vertical range
            # mask = (lines_df['vpos'] >= block_top) & (lines_df['vpos'] + lines_df['height'] <= block_bottom)
            mask = (lines_df['vpos'] >= block_top) & (lines_df['vpos'] <= block_bottom)

            lines_df.loc[mask, 'block_id'] = block['id']

        return lines_df

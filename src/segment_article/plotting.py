from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from io import BytesIO
import requests



def fetch_image(
        page_id, 
        image_cache:str|Path|None=None
    ) -> Image.Image:
    """Fetch an image for the given page ID, optionally using a local cache.

    Args:
        page_id: The identifier of the page to fetch.
        image_cache: Optional directory to use for caching downloaded images.

    Returns:
        PIL.Image.Image: The fetched image.
    """

    if image_cache:
        image_cache = Path(image_cache)
        file = page_id.replace(":", "_") + ".jpg"
        cached_image_path = image_cache / file
        if cached_image_path.exists():
            return Image.open(cached_image_path)
    
    img_url = f"https://resolver.kb.nl/resolve?urn={page_id}:image"

    with requests.get(img_url) as response:
        response.raise_for_status()
        img_content = response.content
        img = Image.open(BytesIO(img_content))
    
    if image_cache:
        image_cache.mkdir(parents=True, exist_ok=True)
        img.save(cached_image_path)

    return img


def plot_blocks(
    image,
    df,
    color_column=None,
    add_fill=True,
    add_id=True,
    add_color_id=True,
    id_column="id",
    outline_width=2,
    use_color_list=True,
    return_thumbnail=True,
    thumbnail_size=(2000, 2000)
):
    """
    Plot blocks on an image based on the provided DataFrame.

    Args:
        image: The PIL.Image.Image object to draw on.
        df: DataFrame containing block information with columns 'hpos', 'vpos', 
            'width', 'height', and optionally color and id columns.
        color_column: Column name to use for coloring blocks.
        add_fill: Whether to fill the blocks with color.
        add_id: Whether to display the id on the blocks.
        add_color_id: Whether to display the color id on the blocks.
        id_column: Column name for the id.
        outline_width: Width of the block outlines.
        use_color_list: Whether to use the predefined color list.
        return_thumbnail: Whether to return a thumbnail of the result. Usefull
            for faster preview, as original scan images can be large and slow 
            to render.
        thumbnail_size: Size of the thumbnail if return_thumbnail is True. Downsizing
            is done proportionally.

    Returns:
        PIL.Image.Image: The image with plotted blocks.
    """
    COLOR_LIST = [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 0, 255),
        (255, 0, 255, 255),
        (0, 255, 255, 255),
        (128, 0, 0, 255),
        (0, 128, 0, 255),
        (0, 0, 128, 255),
        (128, 128, 0, 255),
        (128, 0, 128, 255),
        (0, 128, 128, 255),
    ]
    img_copy = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    default_outline = (255, 0, 0, 255)
    default_fill = (255, 0, 0, 60)

    color_map = {}

    df = df.copy()

    if add_id:
        # Shorten the id column name, by extracting the capital letters and numbers from the id string
        df[id_column] = df[id_column].astype(str).str.replace(r'[a-z]+', '', regex=True)
    if add_color_id:
        # Shorten the color column name, by extracting the capital letters and numbers from the id string
        df[color_column] = df[color_column].astype(str).str.replace(r'[a-z]+', '', regex=True)
    
    if color_column is not None and color_column in df.columns:
        values = df[color_column].dropna().astype(str).unique()
        if use_color_list:
            color_map = {value: COLOR_LIST[i % len(COLOR_LIST)] for i, value in enumerate(values)}
        else:
            n = max(len(values), 1)
            color_map = {
                value: (
                    int((i + 1) * 255 / n),
                    int((i + 1) * 100 / n),
                    int((i + 1) * 200 / n),
                    255,
                )
                for i, value in enumerate(values)
            }

    PRINTED_KEYS = set()
    for _, row in df.iterrows():
        x0 = int(row["hpos"])
        y0 = int(row["vpos"])
        x1 = x0 + int(row["width"])
        y1 = y0 + int(row["height"])

        if color_map:
            color_key = str(row[color_column])
            outline_color = color_map.get(color_key, default_outline)
            fill_color = (outline_color[0], outline_color[1], outline_color[2], 60) if add_fill else None
            
        else:
            outline_color = default_outline
            fill_color = default_fill if add_fill else None

        draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=outline_width, fill=fill_color)
            
        if add_color_id and color_key not in PRINTED_KEYS:
            # Print id left to the rectangle
            draw.text((x0 - 50, y0), color_key, fill="black", font_size=30, stroke_width=2)
            PRINTED_KEYS.add(color_key)

        if add_id and id_column in df.columns and pd.notna(row[id_column]):
            draw.text((x0, y0), str(row[id_column]), fill="black", font_size=30)


    result = Image.alpha_composite(img_copy, overlay)
    
    if return_thumbnail:
        if isinstance(thumbnail_size, int):
            thumbnail_size = (thumbnail_size, thumbnail_size)

        result.thumbnail(thumbnail_size)

    return result
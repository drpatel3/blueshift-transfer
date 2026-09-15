import os
from pathlib import Path
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import logging

import config

# Download required NLTK data (skip if pre-installed, e.g. Lambda container)
if not os.environ.get("NLTK_DATA"):
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.stem import WordNetLemmatizer


# Get English stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
logger = logging.getLogger(__name__)

# Lazy-loaded CLIP model for image vectorization
_clip_model = None

def get_clip_model():
    """Lazy load CLIP model for image embedding."""
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading CLIP model...")
        _clip_model = SentenceTransformer('clip-ViT-B-32')
        logger.info("CLIP model loaded.")
    return _clip_model

def vectorize_image(pil_image):
    """Convert PIL image to 512-dim embedding vector."""
    if not config.ENABLE_CLIP:
        return None
    model = get_clip_model()
    # Handle pdfplumber PageImage wrapper - extract the actual PIL image
    if hasattr(pil_image, 'original'):
        pil_image = pil_image.original
    embedding = model.encode(pil_image)
    return embedding.tolist()  # Convert numpy array to list for JSON serialization


def preprocess_text(text):
    """Remove stopwords and lemmatize."""
    words = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w.lower()) for w in words
                if w.lower() not in stop_words and w.isalpha()]
    return ' '.join(filtered)


def is_flowsheet_caption(text):
    """Check if caption/text indicates a process flowsheet.

    Requires two conditions:
      1. A figure label word (figure, figura, diagram, exhibit, drawing, flow)
      2. A flowsheet term (process flow, flowsheet, block diagram, etc.)
    """
    text = text.lower()
    has_figure = any(kw in text for kw in (
        'figure', 'figura', 'diagram', 'exhibit', 'drawing',
    ))
    has_flowsheet_term = any(kw in text for kw in (
        'process flow', 'flowsheet', 'flow sheet', 'flow-sheet',
        'block flow', 'block diagram', 'simplified flow', 'process diagram',
        'diagrama de flujo', 'diagrama de proceso',
        'plant layout', 'general arrangement', 'circuit diagram',
        'processing plant', 'treatment plant', 'recovery process',
    ))
    return has_figure and has_flowsheet_term


def filter_overlapping_images(images, overlap_threshold=0.5):
    """
    Filter out images that significantly overlap, keeping the larger one.

    Args:
        images: List of image dicts with 'top', 'x0', 'width', 'height' keys
        overlap_threshold: Minimum overlap ratio (0-1) to consider as duplicate

    Returns:
        Filtered list of images with overlapping duplicates removed
    """
    if len(images) <= 1:
        return images

    def get_area(img):
        return img['width'] * img['height']

    def get_overlap_ratio(img1, img2):
        # Calculate bounding boxes
        x1_min, x1_max = img1['x0'], img1['x0'] + img1['width']
        y1_min, y1_max = img1['top'], img1['top'] + img1['height']
        x2_min, x2_max = img2['x0'], img2['x0'] + img2['width']
        y2_min, y2_max = img2['top'], img2['top'] + img2['height']

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0  # No overlap

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        smaller_area = min(get_area(img1), get_area(img2))

        return inter_area / smaller_area if smaller_area > 0 else 0

    # Sort by area (largest first)
    sorted_images = sorted(images, key=get_area, reverse=True)
    kept = []

    for img in sorted_images:
        is_duplicate = False
        for kept_img in kept:
            if get_overlap_ratio(img, kept_img) >= overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(img)

    return kept


def tokenize_text_to_paragraph(text):
    text = text
    if text:
        text = text.strip()
        paragraphs = text.split('\n\n')
        # Remove stopwords from each paragraph
        filtered_paragraphs = []
        for paragraph in paragraphs:
            words = word_tokenize(paragraph)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            if len(filtered_words) > 1:
                filtered_paragraphs.append(' '.join(filtered_words))
        filtered_paragraphs = [p for p in filtered_paragraphs if p.strip() != '']
    
    return filtered_paragraphs


def find_charts_pypdf(pdf_location=None):
    with open(pdf_location, 'rb') as file:
        content = file.read() 
    pdf = PdfReader(test_pdf_path)
    chart_pages = []
    for page_num, page in enumerate(pdf.pages):
        for image_key, image_obj in page.images.items():
            logger.info(f"Found image on page {page_num + 1}: {image_key}")
            chart_pages.append(page_num + 1)
    return chart_pages
    

def extract_content_with_positions(pdf_location=None):
    """Extract text and images with their positions using pdfplumber."""
    import pdfplumber

    results = []

    with pdfplumber.open(pdf_location) as pdf:
        for page_num, page in enumerate(pdf.pages[3:6]):
            page_content = {
                'page': page_num + 1,
                'elements': []
            }

            # Get text with positions (each word has coordinates)
            words = page.extract_words()
            for word in words:
                page_content['elements'].append({
                    'type': 'text',
                    'content': word['text'],
                    'top': word['top'],
                    'bottom': word['bottom'],
                    'x0': word['x0'],
                    'x1': word['x1']
                })

            # Get images with positions
            for img in page.images:
                page_content['elements'].append({
                    'type': 'image',
                    'top': img['top'],
                    'bottom': img['top'] + img['height'],
                    'x0': img['x0'],
                    'x1': img['x0'] + img['width'],
                    'width': img['width'],
                    'height': img['height']
                })

            # Sort all elements by vertical position (top to bottom)
            page_content['elements'].sort(key=lambda e: e['top'])

            results.append(page_content)

    return results


def detect_charts_pdfplumber(pdf_location=None):
    """Detect both embedded images and vector-based charts using pdfplumber."""
    import pdfplumber
    import re

    results = []

    with pdfplumber.open(pdf_location) as pdf:
        for page_num, page in enumerate(pdf.pages[3:7]):
            page_info = {
                'page': page_num + 1,
                'images': [],
                'vector_charts': []
            }

            # Detect embedded images
            for img in page.images:
                page_info['images'].append({
                    'top': img['top'],
                    'x0': img['x0'],
                    'width': img['width'],
                    'height': img['height']
                })

            # Detect vector charts by looking for drawing objects (lines, rects, curves)
            lines = page.lines if hasattr(page, 'lines') else []
            rects = page.rects if hasattr(page, 'rects') else []
            curves = page.curves if hasattr(page, 'curves') else []

            # If there are many drawing objects, likely a vector chart
            drawing_objects = list(lines) + list(rects) + list(curves)

            if len(drawing_objects) > 10:  # threshold for "many" drawing objects
                # Find bounding box of all drawing objects
                if drawing_objects:
                    min_top = min(obj.get('top', obj.get('y0', 0)) for obj in drawing_objects)
                    max_bottom = max(obj.get('bottom', obj.get('y1', 0)) for obj in drawing_objects)
                    min_x = min(obj.get('x0', 0) for obj in drawing_objects)
                    max_x = max(obj.get('x1', 0) for obj in drawing_objects)

                    page_info['vector_charts'].append({
                        'top': min_top,
                        'bottom': max_bottom,
                        'x0': min_x,
                        'x1': max_x,
                        'num_drawing_objects': len(drawing_objects)
                    })

            # Also look for "Figure X" captions and extract full caption text
            words = page.extract_words()
            for i, word in enumerate(words):
                if word['text'].lower() == 'figure' and i + 1 < len(words):
                    next_word = words[i + 1]['text']
                    if re.match(r'^\d+\.?$', next_word):
                        # Get the caption's vertical position
                        caption_top = word['top']

                        # Collect all words on the same line (within similar y position)
                        caption_words = []
                        for w in words:
                            # Words are on the same line if their top is within a few pixels
                            if abs(w['top'] - caption_top) < 5:
                                caption_words.append((w['x0'], w['text']))

                        # Sort by x position and join
                        caption_words.sort(key=lambda x: x[0])
                        full_caption = ' '.join(w[1] for w in caption_words)

                        page_info['vector_charts'].append({
                            'type': 'figure_caption',
                            'label': f"Figure {next_word}",
                            'caption': full_caption,
                            'top': caption_top,
                            'x0': word['x0']
                        })

            if page_info['images'] or page_info['vector_charts']:
                results.append(page_info)

    return results


def extract_content_with_figures(pdf_location=None, page_range=None):
    """
    Extract text and match it with nearby figures/charts.
    Returns structured content with text grouped around figures.
    """
    import pdfplumber
    import re

    results = []

    with pdfplumber.open(pdf_location) as pdf:
        pages = pdf.pages[page_range[0]:page_range[1]] if page_range else pdf.pages

        for page_num, page in enumerate(pages):
            actual_page_num = (page_range[0] if page_range else 0) + page_num + 1

            # Collect all elements with positions
            elements = []

            # Get text lines (group words into lines)
            words = page.extract_words()
            lines = {}
            for word in words:
                # Group words by their top position (within tolerance)
                line_key = round(word['top'] / 5) * 5  # Round to nearest 5 pixels
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append((word['x0'], word['text']))

            # Convert lines dict to sorted text elements
            for line_top, line_words in sorted(lines.items()):
                line_words.sort(key=lambda x: x[0])
                line_text = ' '.join(w[1] for w in line_words)
                elements.append({
                    'type': 'text',
                    'content': line_text,
                    'top': line_top
                })

            # Get embedded images
            for img in page.images:
                elements.append({
                    'type': 'image',
                    'top': img['top'],
                    'bottom': img['top'] + img['height'],
                    'width': img['width'],
                    'height': img['height']
                })

            # Detect figure captions
            for i, word in enumerate(words):
                if word['text'].lower() == 'figure' and i + 1 < len(words):
                    next_word = words[i + 1]['text']
                    if re.match(r'^\d+\.?$', next_word):
                        caption_top = word['top']
                        caption_words = []
                        for w in words:
                            if abs(w['top'] - caption_top) < 5:
                                caption_words.append((w['x0'], w['text']))
                        caption_words.sort(key=lambda x: x[0])
                        full_caption = ' '.join(w[1] for w in caption_words)

                        elements.append({
                            'type': 'figure',
                            'label': f"Figure {next_word}",
                            'caption': full_caption,
                            'top': caption_top
                        })

            # Sort all elements by vertical position
            elements.sort(key=lambda e: e['top'])

            # Build sequential content - text blocks followed by figures, then more text
            grouped_content = []
            current_text_block = []

            for elem in elements:
                if elem['type'] == 'text':
                    # Skip if this line is part of a caption
                    is_caption = any(
                        e['type'] == 'figure' and abs(e['top'] - elem['top']) < 10
                        for e in elements
                    )
                    if not is_caption:
                        current_text_block.append(elem['content'])
                elif elem['type'] in ('figure', 'image'):
                    # First, save any accumulated text as a text block
                    if current_text_block:
                        grouped_content.append({
                            'type': 'text_block',
                            'content': ' '.join(current_text_block),
                            'top': elem['top'] - 1  # Just before the figure
                        })
                        current_text_block = []

                    # Then add the figure/image
                    grouped_content.append({
                        'type': elem['type'],
                        'caption': elem.get('caption', '[Embedded Image]'),
                        'label': elem.get('label', 'Image'),
                        'top': elem['top']
                    })

            # Add any remaining text after the last figure
            if current_text_block:
                grouped_content.append({
                    'type': 'text_block',
                    'content': ' '.join(current_text_block),
                    'top': 9999
                })

            results.append({
                'page': actual_page_num,
                'content': grouped_content
            })

    return results


def extract_and_save_images(pdf_location=None, output_folder=None, page_range=None):
    """
    Extract all images from a PDF and save them to a folder.
    Also captures vector charts as page screenshots.
    """

    import pdfplumber
    import re

    # Set default output folder
    if output_folder is None:
        output_folder = Path(__file__).parent / "extracted_images"

    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    saved_files = []

    # Extract figures and images using pdfplumber
    logger.info("Extracting figures and images with pdfplumber...")
    with pdfplumber.open(pdf_location) as pdf:
        pages = pdf.pages[page_range[0]:page_range[1]] if page_range else pdf.pages

        for page_num, page in enumerate(pages):
            actual_page_num = (page_range[0] if page_range else 0) + page_num + 1
            words = page.extract_words()
            images = page.images

            # Extract embedded images (skip header/footer images)
            header_threshold = page.height * 0.05  # Skip images in top 10% of page
            footer_threshold = page.height * 0.9  # Skip images in bottom 10% of page

            # Debug: show what we're working with
            text_lines = set(round(w['top'] / 10) for w in words)

            # Track saved figures on this page to avoid duplicates
            saved_figures_on_page = set()

            # Check if this is a figure-heavy page (few text lines)
            # If so, save one combined image instead of processing each separately
            non_footer_images = [img for img in images if header_threshold < img['top'] < footer_threshold]
            if len(non_footer_images) > 1 and len(text_lines) <= 15:
                # Multiple images with little text = likely one multi-part figure
                # Find the bounding box of all non-footer images
                all_tops = [img['top'] for img in non_footer_images]
                all_bottoms = [img['top'] + img['height'] for img in non_footer_images]
                combined_top = min(all_tops)
                combined_bottom = max(all_bottoms)

                # Find caption below the combined image area
                caption_text = ""
                for w in words:
                    if combined_bottom < w['top'] < combined_bottom + 100:
                        if w['text'].lower() in ('figure', 'figura', 'diagram', 'exhibit', 'drawing'):
                            caption_line_top = w['top']
                            caption_words = [(cw['x0'], cw['text']) for cw in words if abs(cw['top'] - caption_line_top) < 5]
                            caption_words.sort(key=lambda x: x[0])
                            caption_text = '_'.join(cw[1] for cw in caption_words[:3])
                            combined_bottom = max(w['bottom'] for w in words if abs(w['top'] - caption_line_top) < 5) + 10
                            break

                # Find text above
                text_above = [w for w in words if w['bottom'] < combined_top - 10]
                if text_above:
                    crop_top = max(w['bottom'] for w in text_above) + 5
                else:
                    crop_top = page.height * 0.05

                if caption_text:
                    safe_name = re.sub(r'[^\w\-.]', '_', caption_text)
                    img_label = f"page{page_num + 1}_{safe_name}"
                else:
                    img_label = f"page{page_num + 1}_combined_figure"

                try:
                    cropped = page.crop((0, max(0, crop_top), page.width, min(page.height, combined_bottom)))
                    img_out = cropped.to_image(resolution=150)
                    output_path = output_folder / f"{img_label}.png"
                    img_out.save(str(output_path))
                    saved_files.append(str(output_path))
                except Exception as e:
                    logger.warning(f"  Could not extract combined figure: {e}")
                continue  # Skip individual image processing for this page

            for img_idx, img in enumerate(images):
                img_top = img['top']
                if img_top < header_threshold or img_top > footer_threshold:
                    continue  # Skip header/footer images
                img_bottom = img['top'] + img['height']
                img_left = img['x0']
                img_right = img['x0'] + img['width']

                # Look for a caption below the image (within 100 pixels or near page bottom)
                caption_text = ""
                for w in words:
                    if img_bottom < w['top'] < min(img_bottom + 100, page.height):
                        if w['text'].lower() in ('figure', 'figura', 'diagram', 'exhibit', 'drawing'):
                            # Find the full caption line
                            caption_line_top = w['top']
                            caption_words = []
                            for cw in words:
                                if abs(cw['top'] - caption_line_top) < 5:
                                    caption_words.append((cw['x0'], cw['text']))
                            caption_words.sort(key=lambda x: x[0])
                            caption_text = '_'.join(cw[1] for cw in caption_words[:3])
                            break

                # Create filename
                if caption_text:
                    safe_name = re.sub(r'[^\w\-.]', '_', caption_text)
                    img_label = f"page{page_num + 1}_{safe_name}"
                    # Skip if we already saved this figure on this page
                    if safe_name in saved_figures_on_page:
                        logger.info(f"    Skipping duplicate: {safe_name}")
                        continue
                    saved_figures_on_page.add(safe_name)
                else:
                    img_label = f"page{page_num + 1}_image_{img_idx + 1}"

                # Find the last text above the image to use as top boundary
                text_above_img = [w for w in words if w['bottom'] < img_top - 10]

                # Check if page has minimal text (likely full-page figure)
                # Count distinct text lines (group by y position)
                text_lines = set(round(w['top'] / 10) for w in words)
                is_full_page_figure = len(text_lines) <= 3  # Only caption lines

                if is_full_page_figure:
                    # Take nearly full page with margins
                    crop_top = page.height * 0.05
                    crop_bottom = page.height * 0.95
                else:
                    if text_above_img:
                        crop_top = max(w['bottom'] for w in text_above_img) + 5
                    else:
                        crop_top = 0

                    # Find caption position for bottom boundary
                    if caption_text:
                        caption_words_below = [w for w in words if w['top'] > img_bottom and w['top'] < img_bottom + 100]
                        if caption_words_below:
                            crop_bottom = max(w['bottom'] for w in caption_words_below) + 10
                        else:
                            crop_bottom = img_bottom + 50
                    else:
                        crop_bottom = img_bottom + 10

                # Crop and save the image region
                try:
                    cropped = page.crop((
                        0,
                        max(0, crop_top),
                        page.width,
                        min(page.height, crop_bottom)
                    ))
                    img_out = cropped.to_image(resolution=150)
                    output_path = output_folder / f"{img_label}.png"
                    img_out.save(str(output_path))
                    saved_files.append(str(output_path))
                    logger.info(f"  Saved: {output_path}")
                except Exception as e:
                    logger.warning(f"  Could not extract image: {e}")

            # Find figure captions (for vector charts)
            for i, word in enumerate(words):
                if word['text'].lower() == 'figure' and i + 1 < len(words):
                    next_word = words[i + 1]['text']
                    if re.match(r'^\d+\.?$', next_word):
                        fig_label = f"figure_{next_word.rstrip('.')}"

                        # Get caption position
                        caption_top = word['top']

                        # Check if this caption is already associated with an embedded image
                        is_for_embedded_image = any(
                            img['top'] + img['height'] < caption_top < img['top'] + img['height'] + 50
                            for img in images
                        )

                        if is_for_embedded_image:
                            continue  # Skip - already extracted with the image

                        # Find the last text block above the caption to use as top boundary
                        text_above = [w for w in words if w['top'] < caption_top - 20]
                        if text_above:
                            chart_top = max(w['bottom'] for w in text_above) + 5
                        else:
                            chart_top = 0

                        chart_bottom = caption_top + 25  # Include the caption line

                        # Crop the full width of the page for the figure
                        try:
                            cropped = page.crop((
                                0,
                                max(0, chart_top),
                                page.width,
                                min(page.height, chart_bottom)
                            ))
                            img = cropped.to_image(resolution=150)
                            output_path = output_folder / f"file_page{actual_page_num}_{fig_label}.png"
                            img.save(str(output_path))
                            saved_files.append(str(output_path))
                            logger.info(f"  Saved: {output_path}")
                        except Exception as e:
                            logger.warning(f"  Could not extract {fig_label}: {e}")

    logger.info(f"\nTotal files saved: {len(saved_files)}")
    logger.info(f"Output folder: {output_folder.absolute()}")
    return saved_files


def find_process_flowsheet(pdf_location, output_folder=None, page_range=None):
    """Find and extract all figures captioned 'process flowsheet' from a PDF."""
    import pdfplumber
    from pathlib import Path
    import time
    import re

    if output_folder is None:
        output_folder = Path(__file__).parent / "extracted_images"
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Get PDF name for unique filenames
    pdf_name = Path(pdf_location).stem
    pdf_name = re.sub(r'[^\w\-]', '_', pdf_name)  # sanitize

    found_paths = []
    flowsheet_count = 0
    last_found_time = time.time()

    with pdfplumber.open(pdf_location) as pdf:
        start_page = 120
        for page_num, page in enumerate(pdf.pages[start_page:]):
            # Timeout if no chart found in 60 seconds
            if time.time() - last_found_time > 60:
                logger.warning("Timeout: no flowsheet found in 60 seconds, stopping.")
                break
            actual_page_num = start_page + page_num

            # Handle rotated pages - check if page is landscape/rotated
            is_rotated = page.rotation in (90, 270) or page.width > page.height * 1.3

            words = page.extract_words()
            images = page.images

            # For rotated pages: search ALL page text for flowsheet keywords
            # Don't require images - pdfplumber may not detect them on rotated pages
            if is_rotated:
                all_text = ' '.join(w['text'].lower() for w in words)
                has_flowsheet = 'figure' in all_text and (
                    'process flow' in all_text or 'flowsheet' in all_text or
                    'flow sheet' in all_text or 'block flow' in all_text or
                    'block diagram' in all_text or 'simplified flow' in all_text
                )

                if has_flowsheet:
                    try:
                        # Trim top/bottom 10% to remove headers/footers
                        crop_top = page.height * 0.1
                        crop_bottom = page.height * 0.9
                        cropped = page.crop((0, crop_top, page.width, crop_bottom))
                        img_out = cropped.to_image(resolution=150)
                        flowsheet_count += 1
                        output_path = output_folder / f"{pdf_name}_flowsheet_{flowsheet_count}_page{actual_page_num}.png"
                        img_out.save(str(output_path))
                        if output_path.exists():
                            logger.info(f"Saved (rotated): {output_path}")
                            found_paths.append(str(output_path))
                        else:
                            logger.warning(f"WARNING: File not saved: {output_path}")
                        last_found_time = time.time()
                    except Exception as e:
                        logger.error(f"Could not extract flowsheet on page {actual_page_num}: {e}")
                continue

            if not images:
                continue

            header_threshold = page.height * 0.08
            footer_threshold = page.height * 0.92

            for img in images:
                if img['top'] < header_threshold or img['top'] > footer_threshold:
                    continue

                img_top = img['top']
                img_bottom = img['top'] + img['height']

                # Look for caption ABOVE the image (within 60 pixels)
                caption_above = [w for w in words if img_top - 60 < w['bottom'] < img_top]
                text_above = ' '.join(w['text'].lower() for w in sorted(caption_above, key=lambda w: (w['top'], w['x0'])))

                # Look for caption BELOW the image (within 60 pixels)
                caption_below = [w for w in words if img_bottom < w['top'] < img_bottom + 60]
                text_below = ' '.join(w['text'].lower() for w in sorted(caption_below, key=lambda w: (w['top'], w['x0'])))

                # Check either location for "figure" + flowsheet keywords
                def has_flowsheet_caption(text):
                    return 'figure' in text and (
                        'process flow' in text or 'flowsheet' in text or
                        'flow sheet' in text or 'flow-sheet' in text or
                        'block flow' in text or 'block diagram' in text or
                        'simplified flow' in text
                    )

                if not (has_flowsheet_caption(text_above) or has_flowsheet_caption(text_below)):
                    continue

                # Crop tightly around the image with small margin
                margin = 5
                crop_top = max(0, img_top - margin)
                crop_bottom = min(page.height, img_bottom + margin)
                crop_height = crop_bottom - crop_top
                crop_width = page.width

                # Skip if crop is too rectangular (flowsheets are nearly square)
                aspect_ratio = crop_width / crop_height if crop_height > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 3.0:
                    continue

                try:
                    cropped = page.crop((0, crop_top, page.width, crop_bottom))
                    img_out = cropped.to_image(resolution=150)
                    flowsheet_count += 1
                    output_path = output_folder / f"{pdf_name}_flowsheet_{flowsheet_count}_page{actual_page_num}.png"
                    img_out.save(str(output_path))
                    if output_path.exists():
                        logger.info(f"Saved: {output_path}")
                        found_paths.append(str(output_path))
                    else:
                        logger.warning(f"WARNING: File not saved: {output_path}")
                    last_found_time = time.time()
                except Exception as e:
                    logger.error(f"Could not extract flowsheet on page {actual_page_num}: {e}")

    if not found_paths:
        logger.info("No 'process flowsheet' caption found.")
    else:
        logger.info(f"Found {len(found_paths)} process flowsheet(s) total.")
    return found_paths


def extract_all_figures(pdf_location, output_folder=None, flowsheet_start_page=5, start_page=5, max_aspect_ratio=2.4, vectorize=True, image_types=("flowsheet", "other"), page_filter=None):
    """
    Single-pass extraction of all figures, tagged as 'flowsheet' or 'other'.

    - Extracts images starting from start_page (default 5)
    - Checks for flowsheet captions on all pages from start_page onward
    - Skips images wider than max_aspect_ratio times their height (default 2.4)
    - Uses smart cropping: combines multi-part figures, detects full-page figures
    - Saves flowsheets to pfs/ subfolder, others are vectorized and saved to embeddings.json
    - image_types: tuple of types to extract — ("flowsheet", "other") for all,
      ("flowsheet",) for flowsheets only, ("other",) for non-flowsheets only
    """
    import pdfplumber
    import re
    import time
    import json

    if output_folder is None:
        output_folder = Path(__file__).parent / "extracted_images"
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Create folder for flowsheets only (other images are vectorized, not saved)
    pfs_folder = output_folder / "pfs"
    pfs_folder.mkdir(exist_ok=True)

    # Load existing embeddings or start fresh
    embeddings_path = output_folder / "embeddings.json"
    if embeddings_path.exists():
        try:
            with open(embeddings_path) as f:
                embeddings_data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            embeddings_data = {}
    else:
        embeddings_data = {}

    pdf_name = Path(pdf_location).stem
    pdf_name = re.sub(r'[^\w\-]', '_', pdf_name)
    extracted = []
    last_found_time = time.time()

    with pdfplumber.open(pdf_location) as pdf:
        # When page_filter is provided, only iterate those pages (fast)
        if page_filter is not None:
            pages_to_scan = [(p - 1, pdf.pages[p - 1]) for p in sorted(page_filter)
                             if 0 < p <= len(pdf.pages)]
        else:
            pages_to_scan = list(enumerate(pdf.pages))

        for page_num, page in pages_to_scan:
            # Timeout if no images found in 60 seconds (skip for filtered and flowsheet-only scans)
            if page_filter is None and image_types != ("flowsheet",) and time.time() - last_found_time > 60:
                logger.warning("Timeout: no images found in 60 seconds, stopping.")
                break

            actual_page_num = page_num + 1  # 1-indexed page number

            # Skip pages before start_page
            if actual_page_num < start_page:
                continue

            words = page.extract_words()
            page_text = ' '.join(w['text'] for w in words)

            # Flowsheet-only mode: skip pages without flowsheet keywords
            if image_types == ("flowsheet",):
                if not is_flowsheet_caption(page_text):
                    continue

            images = page.images

            # Handle rotated/landscape pages - pdfplumber may not detect images
            is_rotated = page.rotation in (90, 270) or page.width > page.height * 1.3
            if is_rotated:
                if "flowsheet" not in image_types:
                    continue
                all_text = ' '.join(w['text'].lower() for w in words)
                if is_flowsheet_caption(all_text):
                    try:
                        crop_top = page.height * 0.1
                        crop_bottom = page.height * 0.9
                        cropped = page.crop((0, crop_top, page.width, crop_bottom))
                        pil_image = cropped.to_image(resolution=150)
                        output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}.png"
                        pil_image.save(str(output_path))
                        extracted.append({
                            "type": "flowsheet",
                            "text_before": "",
                            "text_after": "",
                            "caption": all_text[:200],
                            "page": actual_page_num,
                            "image_path": str(output_path)
                        })
                        logger.info(f"  Saved [flowsheet] (rotated): {output_path.name}")
                        last_found_time = time.time()
                    except Exception as e:
                        logger.error(f"  Could not extract rotated flowsheet on page {actual_page_num}: {e}")
                continue  # Skip normal processing for rotated pages

            # Header/footer thresholds
            header_threshold = page.height * 0.10
            footer_threshold = page.height * 0.90

            # Filter out header/footer images and tiny images (likely icons/decorations)
            min_image_size = 100  # minimum width or height in points
            content_images = [img for img in images
                            if header_threshold < img['top'] < footer_threshold
                            and img['width'] >= min_image_size
                            and img['height'] >= min_image_size]

            if not content_images:
                # Vector diagram detection: pages with no meaningful raster images
                # (tiny logos/watermarks are filtered out above)
                if "flowsheet" in image_types:
                    all_text = ' '.join(w['text'].lower() for w in words)
                    has_drawings = (len(page.lines or []) + len(page.rects or []) + len(page.curves or [])) > 30
                    if has_drawings and is_flowsheet_caption(all_text):
                        try:
                            crop_top = page.height * 0.05
                            crop_bottom = page.height * 0.95
                            cropped = page.crop((0, crop_top, page.width, crop_bottom))
                            pil_image = cropped.to_image(resolution=150)
                            output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}.png"
                            pil_image.save(str(output_path))
                            extracted.append({
                                "type": "flowsheet",
                                "text_before": "",
                                "text_after": "",
                                "caption": all_text[:200],
                                "page": actual_page_num,
                                "image_path": str(output_path)
                            })
                            logger.info(f"  Saved [flowsheet] (vector): {output_path.name}")
                            last_found_time = time.time()
                        except Exception as e:
                            logger.error(f"  Could not extract vector diagram on page {actual_page_num}: {e}")
                continue

            # Count distinct text lines to detect figure-heavy pages
            text_lines = set(round(w['top'] / 10) for w in words)

            # Filter out overlapping images (keep only the largest)
            before_count = len(content_images)
            content_images = filter_overlapping_images(content_images)
            if before_count != len(content_images):
                logger.info(f"  Page {actual_page_num}: filtered {before_count} -> {len(content_images)} images")

            # Track counts for naming
            page_flowsheet_count = 0
            page_other_count = 0
            saved_figures_on_page = set()

            # Check if this is a multi-part figure page (many images = likely tiled/composite figure)
            # Combine into one image if: many images OR (multiple images with few text lines)
            if len(content_images) > 5 or (len(content_images) > 1 and len(text_lines) <= 15):
                # Combine all images into one
                all_tops = [img['top'] for img in content_images]
                all_bottoms = [img['top'] + img['height'] for img in content_images]
                combined_top = min(all_tops)
                combined_bottom = max(all_bottoms)

                # Find caption below the combined image area
                caption = ""
                for w in words:
                    if combined_bottom < w['top'] < combined_bottom + 100:
                        if w['text'].lower() in ('figure', 'figura', 'diagram', 'exhibit', 'drawing'):
                            caption_line_top = w['top']
                            caption_words = [(cw['x0'], cw['text']) for cw in words if abs(cw['top'] - caption_line_top) < 5]
                            caption_words.sort(key=lambda x: x[0])
                            caption = ' '.join(cw[1] for cw in caption_words)
                            combined_bottom = max(w['bottom'] for w in words if abs(w['top'] - caption_line_top) < 5) + 10
                            break

                # Find text above for top boundary
                text_above = [w for w in words if w['bottom'] < combined_top - 10]
                if text_above:
                    crop_top = max(w['bottom'] for w in text_above) + 5
                else:
                    crop_top = page.height * 0.05

                # Determine type - check caption AND all text near image bounds
                nearby_above = [w for w in words if combined_top - 60 < w['bottom'] < combined_top]
                nearby_below = [w for w in words if combined_bottom < w['top'] < combined_bottom + 60]
                nearby_text = ' '.join(w['text'].lower() for w in sorted(nearby_above + nearby_below, key=lambda w: (w['top'], w['x0'])))
                if is_flowsheet_caption(caption) or is_flowsheet_caption(nearby_text) or is_flowsheet_caption(page_text):
                    fig_type = "flowsheet"
                    page_flowsheet_count += 1
                    output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}.png"
                else:
                    fig_type = "other"
                    page_other_count += 1

                if fig_type not in image_types:
                    continue

                # Get surrounding text
                text_before_words = [w for w in words if crop_top - 200 < w['bottom'] < crop_top - 10]
                raw_before = ' '.join(w['text'] for w in sorted(text_before_words, key=lambda w: (w['top'], w['x0'])))
                text_after_words = [w for w in words if combined_bottom + 10 < w['top'] < combined_bottom + 260]
                raw_after = ' '.join(w['text'] for w in sorted(text_after_words, key=lambda w: (w['top'], w['x0'])))

                try:
                    # Check if image is too wide - if so, make it square using width as driver
                    crop_height = combined_bottom - crop_top
                    if crop_height > 0 and page.width / crop_height > max_aspect_ratio:
                        # Make it square: target height = page width
                        target_height = page.width
                        # Center vertically around the content
                        content_center = (crop_top + combined_bottom) / 2
                        crop_top = content_center - target_height / 2
                        combined_bottom = content_center + target_height / 2
                        # Ensure we stay within page bounds
                        if crop_top < 0:
                            crop_top = 0
                            combined_bottom = target_height
                        if combined_bottom > page.height:
                            combined_bottom = page.height
                            crop_top = max(0, page.height - target_height)

                    cropped = page.crop((0, max(0, crop_top), page.width, min(page.height, combined_bottom)))
                    pil_image = cropped.to_image(resolution=150)

                    # Build result entry
                    result_entry = {
                        "type": fig_type,
                        "text_before": preprocess_text(raw_before),
                        "text_after": preprocess_text(raw_after),
                        "caption": caption,
                        "page": actual_page_num
                    }

                    if fig_type == "flowsheet":
                        # Save flowsheets to disk
                        pil_image.save(str(output_path))
                        result_entry["image_path"] = str(output_path)
                        logger.info(f"  Saved [{fig_type}] (combined): {output_path.name}")
                    elif vectorize == True:
                        # Vectorize "other" images and save to embeddings.json
                        embedding = vectorize_image(pil_image)
                        result_entry["embedding"] = embedding
                        emb_key = f"{pdf_name}_page_{actual_page_num}"
                        embeddings_data[emb_key] = {
                            "embedding": embedding,
                            "text_before": result_entry["text_before"],
                            "text_after": result_entry["text_after"],
                            "caption": caption,
                            "page": actual_page_num
                        }
                        with open(embeddings_path, 'w') as f:
                            json.dump(embeddings_data, f, indent=2)
                        logger.info(f"  Vectorized [{fig_type}] (combined): page {actual_page_num}")

                    extracted.append(result_entry)
                    last_found_time = time.time()
                except Exception as e:
                    logger.error(f"  Could not extract combined figure on page {actual_page_num}: {e}")
                continue  # Skip individual processing for this page

            # Process individual images
            for img_idx, img in enumerate(content_images):
                img_top = img['top']
                img_bottom = img['top'] + img['height']

                # Get caption below image (within 100 pixels)
                caption = ""
                caption_bottom = img_bottom
                for w in words:
                    if img_bottom < w['top'] < min(img_bottom + 100, page.height):
                        if w['text'].lower() in ('figure', 'figura', 'diagram', 'exhibit', 'drawing'):
                            caption_line_top = w['top']
                            caption_words = [(cw['x0'], cw['text']) for cw in words if abs(cw['top'] - caption_line_top) < 5]
                            caption_words.sort(key=lambda x: x[0])
                            caption = ' '.join(cw[1] for cw in caption_words)
                            # Extend bottom to include caption
                            caption_bottom = max(w['bottom'] for w in words if abs(w['top'] - caption_line_top) < 5) + 10
                            break

                # Also check caption above image if none found below
                if not caption:
                    for w in words:
                        if img_top - 60 < w['bottom'] < img_top:
                            if w['text'].lower() in ('figure', 'figura', 'diagram', 'exhibit', 'drawing'):
                                caption_line_top = w['top']
                                caption_words = [(cw['x0'], cw['text']) for cw in words if abs(cw['top'] - caption_line_top) < 5]
                                caption_words.sort(key=lambda x: x[0])
                                caption = ' '.join(cw[1] for cw in caption_words)
                                break

                # Avoid duplicates based on caption
                caption_key = re.sub(r'[^\w]', '', caption.lower()) if caption else f"img{img_idx}"
                if caption_key in saved_figures_on_page:
                    continue
                saved_figures_on_page.add(caption_key)

                # Determine if full-page figure (minimal text)
                is_full_page_figure = len(text_lines) <= 3

                # Calculate crop boundaries
                if is_full_page_figure:
                    crop_top = page.height * 0.05
                    crop_bottom = page.height * 0.95
                else:
                    # Top boundary: last text above image
                    text_above_img = [w for w in words if w['bottom'] < img_top - 10]
                    if text_above_img:
                        crop_top = max(w['bottom'] for w in text_above_img) + 5
                    else:
                        crop_top = max(0, img_top - 10)

                    # Bottom boundary: caption or image bottom
                    crop_bottom = caption_bottom if caption else img_bottom + 10

                # Determine type - check caption AND all text near image bounds
                nearby_above = [w for w in words if img_top - 60 < w['bottom'] < img_top]
                nearby_below = [w for w in words if img_bottom < w['top'] < img_bottom + 60]
                nearby_text = ' '.join(w['text'].lower() for w in sorted(nearby_above + nearby_below, key=lambda w: (w['top'], w['x0'])))
                if is_flowsheet_caption(caption) or is_flowsheet_caption(nearby_text) or is_flowsheet_caption(page_text):
                    fig_type = "flowsheet"
                    page_flowsheet_count += 1
                    if page_flowsheet_count == 1:
                        output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}.png"
                    else:
                        output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}_{page_flowsheet_count}.png"
                else:
                    fig_type = "other"
                    page_other_count += 1

                if fig_type not in image_types:
                    continue

                # Get surrounding text
                text_before_words = [w for w in words if crop_top - 200 < w['bottom'] < crop_top - 10]
                raw_before = ' '.join(w['text'] for w in sorted(text_before_words, key=lambda w: (w['top'], w['x0'])))
                text_after_words = [w for w in words if crop_bottom + 10 < w['top'] < crop_bottom + 260]
                raw_after = ' '.join(w['text'] for w in sorted(text_after_words, key=lambda w: (w['top'], w['x0'])))

                try:
                    # Check if image is too wide - if so, make it square using width as driver
                    crop_height = crop_bottom - crop_top
                    if crop_height > 0 and page.width / crop_height > max_aspect_ratio:
                        # Make it square: target height = page width
                        target_height = page.width
                        # Center vertically around the content
                        content_center = (crop_top + crop_bottom) / 2
                        crop_top = content_center - target_height / 2
                        crop_bottom = content_center + target_height / 2
                        # Ensure we stay within page bounds
                        if crop_top < 0:
                            crop_top = 0
                            crop_bottom = target_height
                        if crop_bottom > page.height:
                            crop_bottom = page.height
                            crop_top = max(0, page.height - target_height)

                    cropped = page.crop((0, max(0, crop_top), page.width, min(page.height, crop_bottom)))
                    pil_image = cropped.to_image(resolution=150)

                    # Build result entry
                    result_entry = {
                        "type": fig_type,
                        "text_before": preprocess_text(raw_before),
                        "text_after": preprocess_text(raw_after),
                        "caption": caption,
                        "page": actual_page_num
                    }

                    if fig_type == "flowsheet":
                        # Save flowsheets to disk
                        pil_image.save(str(output_path))
                        result_entry["image_path"] = str(output_path)
                        logger.info(f"  Saved [{fig_type}]: {output_path.name}")
                    elif vectorize == True:
                        # Vectorize "other" images and save to embeddings.json
                        # pass
                        embedding = vectorize_image(pil_image)
                        result_entry["embedding"] = embedding
                        emb_key = f"{pdf_name}_page_{actual_page_num}_{page_other_count}"
                        embeddings_data[emb_key] = {
                            "embedding": embedding,
                            "text_before": result_entry["text_before"],
                            "text_after": result_entry["text_after"],
                            "caption": caption,
                            "page": actual_page_num
                        }
                        with open(embeddings_path, 'w') as f:
                            json.dump(embeddings_data, f, indent=2)
                        logger.info(f"  Vectorized [{fig_type}]: page {actual_page_num}")

                    extracted.append(result_entry)
                    last_found_time = time.time()
                except Exception as e:
                    logger.error(f"  Could not extract image on page {actual_page_num}: {e}")

    logger.info(f"\nExtracted {len(extracted)} total images ({len([x for x in extracted if x['type'] == 'flowsheet'])} flowsheets)")
    return extracted


def find_flowsheets_in_sections(pdf_location, output_folder=None, sections=("13", "17"), start_page=50):
    """Pass 2: Extract large images from sections 13/17 regardless of captions.

    Uses pypdf to locate section boundaries, then pdfplumber to extract
    any image larger than 200x200 points from those pages.
    """
    import pdfplumber
    import re
    from table_parser import find_multiple_sections

    if output_folder is None:
        output_folder = Path(__file__).parent / "extracted_images"
    output_folder = Path(output_folder)
    pfs_folder = output_folder / "pfs"
    pfs_folder.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_location).stem
    pdf_name = re.sub(r'[^\w\-]', '_', pdf_name)
    extracted = []

    # Find page ranges for target sections (single PDF read)
    found = find_multiple_sections(str(pdf_location), sections, start_page=start_page)
    page_ranges = [(sec, start, end) for sec, (start, end) in found.items()]
    for sec, start, end in page_ranges:
        logger.info(f"  Section {sec}: pages {start+1}-{end}")

    if not page_ranges:
        logger.info("  No section 13/17 boundaries found")
        return extracted

    with pdfplumber.open(str(pdf_location)) as pdf:
        for sec, sec_start, sec_end in page_ranges:
            for page_idx in range(sec_start, min(sec_end, len(pdf.pages))):
                page = pdf.pages[page_idx]
                actual_page_num = page_idx + 1
                images = page.images
                min_size = 200

                # Filter to large images, keep only the largest per page
                large_images = [img for img in images
                                if img['width'] >= min_size and img['height'] >= min_size]

                if not large_images:
                    continue
                largest = max(large_images, key=lambda i: i['width'] * i['height'])

                for img in [largest]:
                    try:
                        img_top = img['top']
                        img_bottom = img['top'] + img['height']
                        crop_top = max(0, img_top - 10)
                        crop_bottom = min(page.height, img_bottom + 10)
                        cropped = page.crop((0, crop_top, page.width, crop_bottom))
                        pil_image = cropped.to_image(resolution=150)
                        output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}.png"
                        pil_image.save(str(output_path))

                        words = page.extract_words()
                        text_before = ' '.join(w['text'] for w in words if w['bottom'] < img_top)
                        text_after = ' '.join(w['text'] for w in words if w['top'] > img_bottom)

                        extracted.append({
                            "type": "flowsheet",
                            "text_before": preprocess_text(text_before[:500]),
                            "text_after": preprocess_text(text_after[:500]),
                            "caption": f"Section {sec} image",
                            "page": actual_page_num,
                            "image_path": str(output_path),
                            "detection_pass": "section_bounded"
                        })
                        logger.info(f"  Saved [flowsheet] (section {sec}): {output_path.name}")
                    except Exception as e:
                        logger.error(f"  Could not extract section image on page {actual_page_num}: {e}")

    # If too many images found, use CLIP to filter to actual flowsheets
    if len(extracted) > 5 and config.ENABLE_CLIP:
        logger.info(f"  {len(extracted)} images found — running CLIP filter")
        try:
            import numpy as np
            model = get_clip_model()
            ref_emb = model.encode(
                "a mineral processing plant flowsheet diagram showing equipment stages connected by flow arrows"
            )
            scored = []
            for entry in extracted:
                try:
                    from PIL import Image
                    img = Image.open(entry["image_path"])
                    img_emb = model.encode(img)
                    cos_sim = float(np.dot(ref_emb, img_emb) / (
                        np.linalg.norm(ref_emb) * np.linalg.norm(img_emb)))
                    scored.append((cos_sim, entry))
                    logger.info(f"    Page {entry['page']}: CLIP sim={cos_sim:.3f}")
                except Exception as e:
                    logger.warning(f"    Page {entry['page']}: CLIP error — {e}")
                    scored.append((0.0, entry))

            # Keep images above threshold
            clip_threshold = 0.25
            filtered = [entry for sim, entry in scored if sim >= clip_threshold]
            if filtered:
                logger.info(f"  CLIP filter: {len(extracted)} -> {len(filtered)} flowsheet(s)")
                extracted = filtered
            else:
                # If nothing passes threshold, keep the top 3 by similarity
                scored.sort(key=lambda x: x[0], reverse=True)
                extracted = [entry for _, entry in scored[:3]]
                logger.info(f"  CLIP filter: none above threshold, keeping top 3")
        except Exception as e:
            logger.warning(f"  CLIP filter failed: {e} — keeping all {len(extracted)}")

    logger.info(f"  Section extraction found {len(extracted)} flowsheet(s)")
    return extracted


def classify_images_with_clip(pdf_location, output_folder=None, sections=("13", "17"), start_page=50):
    """Pass 3: CLIP visual classification of images in sections 13/17.

    Extracts all large images from target sections and uses CLIP similarity
    to identify flowsheet diagrams visually. Only called when text-based
    methods found nothing.
    """
    import pdfplumber
    import re
    import numpy as np
    from table_parser import find_multiple_sections

    if not config.ENABLE_CLIP:
        logger.info("  CLIP disabled, skipping visual classification")
        return []

    if output_folder is None:
        output_folder = Path(__file__).parent / "extracted_images"
    output_folder = Path(output_folder)
    pfs_folder = output_folder / "pfs"
    pfs_folder.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_location).stem
    pdf_name = re.sub(r'[^\w\-]', '_', pdf_name)

    # Find sections (single PDF read)
    found = find_multiple_sections(str(pdf_location), sections, start_page=start_page)
    page_ranges = [(sec, start, end) for sec, (start, end) in found.items()]

    if not page_ranges:
        return []

    # Load CLIP and encode reference prompt
    try:
        model = get_clip_model()
        ref_emb = model.encode(
            "a mineral processing plant flowsheet diagram showing equipment stages connected by flow arrows"
        )
    except Exception as e:
        logger.warning(f"  CLIP model unavailable: {e}")
        return []

    extracted = []
    threshold = 0.25  # cosine similarity threshold

    with pdfplumber.open(str(pdf_location)) as pdf:
        for sec, sec_start, sec_end in page_ranges:
            for page_idx in range(sec_start, min(sec_end, len(pdf.pages))):
                page = pdf.pages[page_idx]
                actual_page_num = page_idx + 1
                images = page.images
                min_size = 150

                large_images = [img for img in images
                                if img['width'] >= min_size and img['height'] >= min_size]

                for img in large_images:
                    try:
                        img_top = img['top']
                        img_bottom = img['top'] + img['height']
                        crop_top = max(0, img_top - 10)
                        crop_bottom = min(page.height, img_bottom + 10)
                        cropped = page.crop((0, crop_top, page.width, crop_bottom))
                        pil_image = cropped.to_image(resolution=150)

                        # CLIP classification
                        pil_img = pil_image.original if hasattr(pil_image, 'original') else pil_image
                        img_emb = model.encode(pil_img)
                        cos_sim = float(np.dot(ref_emb, img_emb) / (
                            np.linalg.norm(ref_emb) * np.linalg.norm(img_emb)))

                        if cos_sim >= threshold:
                            output_path = pfs_folder / f"{pdf_name}_pfs_{actual_page_num}.png"
                            pil_image.save(str(output_path))
                            extracted.append({
                                "type": "flowsheet",
                                "text_before": "",
                                "text_after": "",
                                "caption": f"CLIP match (sim={cos_sim:.3f})",
                                "page": actual_page_num,
                                "image_path": str(output_path),
                                "detection_pass": "clip_visual"
                            })
                            logger.info(f"  Saved [flowsheet] (CLIP sim={cos_sim:.3f}): {output_path.name}")
                    except Exception as e:
                        logger.error(f"  CLIP classification error on page {actual_page_num}: {e}")

    logger.info(f"  CLIP classification found {len(extracted)} flowsheet(s)")
    return extracted


def extract_flowsheets_ensemble(pdf_location, output_folder=None):
    """Two-pass flowsheet detection.

    Pass 1: Caption keyword matching across all pages (from page 5 onward).
            Searches for flowsheet terms in page text — no section scoping needed.
    Pass 2: If Pass 1 found nothing, CLIP visual classification on sections 13/17.

    Each pass only runs if previous passes found nothing.
    """
    from pypdf import PdfReader

    logger.info("Ensemble flowsheet detection:")

    # Read PDF once for page count
    try:
        reader = PdfReader(str(pdf_location))
        page_count = len(reader.pages)
        if page_count < 50:
            logger.info(f"  Skipping — only {page_count} pages (minimum 50)")
            return []
        logger.info(f"  PDF has {page_count} pages")
    except Exception as e:
        logger.error(f"  Could not read PDF: {e}")
        return []

    # Pass 1: keyword matching across all pages
    logger.info(f"  Pass 1: Caption keyword matching across all pages")
    results = extract_all_figures(
        pdf_location, output_folder=output_folder, image_types=("flowsheet",),
    )
    flowsheets = [x for x in results if x['type'] == 'flowsheet']
    if flowsheets:
        logger.info(f"  Pass 1 found {len(flowsheets)} flowsheet(s) — done")
        return results

    # Pass 2: CLIP visual classification on sections 13/17
    logger.info("  Pass 2: CLIP visual classification (sections 13/17)")
    clip_results = classify_images_with_clip(
        pdf_location, output_folder=output_folder, start_page=50
    )
    if clip_results:
        logger.info(f"  Pass 2 found {len(clip_results)} flowsheet(s) — done")
        return clip_results

    logger.warning("  All passes found 0 flowsheets")
    return []


def extract_section_tables(pdf_location, section="13", start_page=50):
    """
    Fast table extraction using only pypdf (no pdfplumber).

    Scans pages for "Table {section}-X" captions, then parses table
    structure directly from the extracted text.

    Args:
        pdf_location: path to PDF
        section: section number to match (default "13")
        start_page: PDF page index to start scanning from (default 50)

    Returns: list of dicts compatible with extract_tables() output:
        [{table_id, page, table_index, section_number, section_title,
          caption, headers, rows, text_before, text_after}, ...]
    """
    import re

    pdf_name = Path(pdf_location).stem
    pdf_name = re.sub(r'[^\w\-]', '_', pdf_name)
    reader = PdfReader(pdf_location)
    extracted = []

    # Pattern to find table captions like "Table 13-1:" or "Table 13.2:"
    caption_pattern = re.compile(
        rf'^(Table\s+{re.escape(section)}[\-\.]\d+[a-z]?\s*:\s*.+)',
        re.IGNORECASE | re.MULTILINE
    )
    # Pattern to detect we've left section 13 (hit section 14+)
    next_section = int(section) + 1
    exit_pattern = re.compile(
        rf'^\s*{next_section}[\.\s]', re.MULTILINE
    )

    logger.info(f"  Scanning {len(reader.pages)} pages from index {start_page}...")

    # Phase 1: find pages with Table 13 captions
    table_pages = []  # list of (page_index, caption, full_text)
    for i in range(start_page, len(reader.pages)):
        text = reader.pages[i].extract_text() or ""

        # Check for table captions on this page
        for match in caption_pattern.finditer(text):
            table_pages.append((i, match.group(1).strip(), text))

        # Stop scanning once we're past our section
        if exit_pattern.search(text) and not caption_pattern.search(text):
            break

    logger.info(f"  Found {len(table_pages)} table caption(s)")

    # Phase 2: parse table structure from text
    for tbl_idx, (page_idx, caption, page_text) in enumerate(table_pages):
        lines = page_text.split('\n')

        # Find where the caption is in the lines
        caption_line_idx = None
        for li, line in enumerate(lines):
            if caption.split(':')[0] in line:
                caption_line_idx = li
                break

        if caption_line_idx is None:
            continue

        # Collect text before the caption (for context)
        text_before = ' '.join(
            l.strip() for l in lines[max(0, caption_line_idx - 3):caption_line_idx]
            if l.strip()
        )

        # Parse lines after caption into header + data rows
        # Skip blank lines and sub-header lines (e.g., "Assay (% or g/t)")
        data_lines = []
        header_line = None
        for line in lines[caption_line_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                # Blank line after data rows = end of table
                if data_lines:
                    break
                continue

            # Detect header: line with mostly non-numeric short tokens
            tokens = stripped.split()
            numeric_count = sum(
                1 for t in tokens
                if re.match(r'^[\d.<>%,\-]+$', t)
            )

            if header_line is None:
                # Still looking for the header row — it's the line where
                # tokens are mostly short labels (like %Cu, Au, Ag)
                # and not a sub-header like "Assay (% or g/t)"
                if len(tokens) >= 3 and '(' not in stripped:
                    header_line = stripped
                continue

            # Data rows: must have at least some numeric content
            if numeric_count >= 2:
                data_lines.append(stripped)
            elif data_lines:
                # Non-numeric line after data = end of table
                break

        if not header_line:
            continue

        # Split header into columns
        headers = header_line.split()

        # Parse data rows, handling multi-word first column
        rows = []
        num_cols = len(headers)
        for dl in data_lines:
            tokens = dl.split()
            if len(tokens) >= num_cols:
                # Right-align: last N tokens are data, rest is the label
                label = ' '.join(tokens[:len(tokens) - num_cols + 1])
                values = tokens[len(tokens) - num_cols + 1:]
                rows.append([label] + values)
            else:
                # Fewer tokens than headers — just pad
                rows.append(tokens + [''] * (num_cols - len(tokens)))

        # Collect text after the table
        # Find where data ends in the original lines
        last_data_line_idx = caption_line_idx + 1
        for li, line in enumerate(lines[caption_line_idx + 1:], caption_line_idx + 1):
            if data_lines and data_lines[-1] in line:
                last_data_line_idx = li
                break

        text_after = ' '.join(
            l.strip() for l in lines[last_data_line_idx + 1:last_data_line_idx + 4]
            if l.strip()
        )

        # Check continuation on next page
        if page_idx + 1 < len(reader.pages):
            next_text = reader.pages[page_idx + 1].extract_text() or ""
            next_lines = next_text.split('\n')

            # If next page has same table structure (no new Table caption at top)
            # and starts with data rows matching our column count
            has_new_caption = caption_pattern.search(next_text.split('\n')[0]
                                                     if next_lines else "")
            if not has_new_caption:
                for nl in next_lines:
                    nl_stripped = nl.strip()
                    if not nl_stripped:
                        continue
                    tokens = nl_stripped.split()
                    numeric_count = sum(
                        1 for t in tokens if re.match(r'^[\d.<>%,\-]+$', t)
                    )
                    # Skip page headers (SRK Consulting, etc.)
                    if numeric_count < 2:
                        # Check if it's a repeated header line
                        if nl_stripped == header_line:
                            continue
                        # If we haven't found any continuation data yet, skip
                        continue
                    # This looks like a data row
                    if len(tokens) >= num_cols:
                        label = ' '.join(tokens[:len(tokens) - num_cols + 1])
                        values = tokens[len(tokens) - num_cols + 1:]
                        rows.append([label] + values)
                    elif len(tokens) > 2:
                        rows.append(tokens + [''] * (num_cols - len(tokens)))

        table_id = f"{pdf_name}_table_{page_idx + 1}_{tbl_idx}"

        extracted.append({
            "table_id": table_id,
            "page": page_idx + 1,
            "table_index": tbl_idx,
            "section_number": section,
            "section_title": caption,
            "caption": caption,
            "headers": headers,
            "rows": rows,
            "text_before": text_before,
            "text_after": text_after,
        })

    logger.info(f"  Extracted {len(extracted)} tables from section {section}")
    return extracted


def detect_section_boundaries(pdf_location):
    """
    Detect NI 43-101 section boundaries in a PDF using Table/Figure numbering.

    Scans for "Table X-Y" and "Figure X-Y" references to determine where
    each section's content begins, filtering out TOC/List-of-Tables pages.

    Args:
        pdf_location: path to PDF file

    Returns:
        Callable that maps page_number (1-indexed) to (section_number, section_title).
        Returns (None, None) for pages before any detected section.
    """
    import re
    from collections import defaultdict

    _tf_pattern = re.compile(r'(?:Table|Figure)\s+(\d{1,2})[\.\-]', re.IGNORECASE)

    _SECTION_TITLES = {
        2: "Introduction", 3: "Reliance on Other Experts",
        4: "Property Description and Location", 5: "Accessibility, Climate, Local Resources",
        6: "History", 7: "Geological Setting and Mineralisation", 8: "Deposit Types",
        9: "Exploration", 10: "Drilling", 11: "Sample Preparation, Analyses and Security",
        12: "Data Verification", 13: "Mineral Processing and Metallurgical Testing",
        14: "Mineral Resource Estimates", 15: "Mineral Reserve Estimates",
        16: "Mining Methods", 17: "Recovery Methods",
        18: "Project Infrastructure", 19: "Market Studies and Contracts",
        20: "Environmental Studies, Permitting and Social Impact",
        21: "Capital and Operating Costs", 22: "Economic Analysis",
        23: "Adjacent Properties", 24: "Other Relevant Data and Information",
        25: "Interpretation and Conclusions", 26: "Recommendations", 27: "References",
    }

    prescan_reader = PdfReader(pdf_location)

    # Pass 1: collect ALL pages where each section is mentioned (after page 10)
    _sec_all_pages = defaultdict(list)
    for pg_idx, pg in enumerate(prescan_reader.pages):
        page_1 = pg_idx + 1
        if page_1 <= 10:
            continue
        text = pg.extract_text() or ""
        for m in _tf_pattern.finditer(text):
            sec = int(m.group(1))
            if 2 <= sec <= 27:
                _sec_all_pages[sec].append(page_1)

    # Pass 2: detect end of TOC / "List of Tables" cluster
    _toc_end = 10
    if _sec_all_pages:
        page_sec_count = defaultdict(set)
        for sec, pages in _sec_all_pages.items():
            for p in pages:
                page_sec_count[p].add(sec)
        toc_pages = {p for p, secs in page_sec_count.items() if len(secs) >= 3}
        if toc_pages:
            _toc_end = max(toc_pages)

    # Pass 3: first occurrence of each section AFTER the TOC pages
    section_first_page = {}
    for sec, pages in _sec_all_pages.items():
        content_pages = [p for p in pages if p > _toc_end]
        if content_pages:
            section_first_page[sec] = min(content_pages)

    _boundaries = sorted(
        [(pg, str(sec), _SECTION_TITLES.get(sec, f"Section {sec}"))
         for sec, pg in section_first_page.items()],
        key=lambda x: x[0],
    )

    def section_for_page(page_num):
        """Return (section_number, section_title) for a given 1-indexed page."""
        result = (None, None)
        for start_pg, sec_num, sec_title in _boundaries:
            if start_pg <= page_num:
                result = (sec_num, sec_title)
            else:
                break
        return result

    return section_for_page


def extract_page_text(pdf_location, start_page=5):
    """
    Extract full text from every page of a PDF with NI 43-101 section assignment.

    Args:
        pdf_location: path to PDF file
        start_page: skip front matter pages before this (default 5)

    Returns:
        list of dicts: [{page_number, text, section_number, section_title, char_count}, ...]
    """
    section_for_page = detect_section_boundaries(pdf_location)
    reader = PdfReader(pdf_location)
    pages = []

    for pg_idx, page in enumerate(reader.pages):
        page_num = pg_idx + 1
        if page_num < start_page:
            continue
        text = page.extract_text() or ""
        sec_num, sec_title = section_for_page(page_num)
        pages.append({
            "page_number": page_num,
            "text": text,
            "section_number": sec_num,
            "section_title": sec_title,
            "char_count": len(text),
        })

    non_empty = sum(1 for p in pages if p["char_count"] > 0)
    logger.info(f"Extracted text from {len(pages)} pages ({non_empty} non-empty)")
    return pages


def extract_tables(pdf_location, start_page=5, target_section=None):
    """
    Extract all tables from a PDF with surrounding text context.

    Args:
        pdf_location: path to PDF
        start_page: skip front matter (default 5)
        target_section: if set (e.g., "13"), only return tables from that section

    Returns: list of dicts:
        [{table_id, page, section_number, section_title, caption,
          headers, rows, text_before, text_after}, ...]
    """
    import pdfplumber
    import re

    pdf_name = Path(pdf_location).stem
    pdf_name = re.sub(r'[^\w\-]', '_', pdf_name)
    extracted = []

    _section_for_page = detect_section_boundaries(pdf_location)

    # --- Fast pre-scan with pypdf to find candidate pages ---
    candidate_pages = None  # None = scan all pages (no target_section)
    target_pattern = None
    if target_section is not None:
        target_pattern = re.compile(
            rf'\btable\s+{re.escape(str(target_section))}[\s\-\.]',
            re.IGNORECASE
        )
        reader = PdfReader(pdf_location)
        candidate_pages = set()
        for i, pg in enumerate(reader.pages):
            text = pg.extract_text() or ""
            if target_pattern.search(text):
                candidate_pages.add(i)  # 0-indexed
        logger.info(f"  pypdf pre-scan: {len(candidate_pages)} candidate page(s) "
              f"(of {len(reader.pages)})")
        if not candidate_pages:
            logger.info(f"\nExtracted 0 tables matching 'Table {target_section}*'")
            return extracted

    # When we have candidate pages, write only those to a temp PDF so
    # pdfplumber doesn't parse the entire (potentially 300+ page) document.
    import tempfile
    temp_pdf_path = None
    page_mapping = {}  # temp_index -> original_0indexed_page

    if candidate_pages is not None:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for temp_idx, orig_idx in enumerate(sorted(candidate_pages)):
            writer.add_page(reader.pages[orig_idx])
            page_mapping[temp_idx] = orig_idx
        tmp_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf')
        os.close(tmp_fd)
        with open(temp_pdf_path, 'wb') as tmp:
            writer.write(tmp)

    pdf_to_open = temp_pdf_path if temp_pdf_path else pdf_location

    with pdfplumber.open(pdf_to_open) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Map back to original page number
            if candidate_pages is not None:
                orig_page_0 = page_mapping[page_num]
                actual_page_num = orig_page_0 + 1
            else:
                actual_page_num = page_num + 1

            if actual_page_num < start_page:
                continue

            # --- Find tables on this page ---
            try:
                tables_on_page = page.find_tables()
            except Exception:
                continue

            if not tables_on_page:
                continue

            # Only extract words on pages we'll actually process
            words = page.extract_words()

            # Look up section from pre-scanned boundary map
            current_section, current_section_title = _section_for_page(actual_page_num)

            for tbl_idx, table_obj in enumerate(tables_on_page):
                try:
                    from table_parser import parse_table
                    raw_rows = table_obj.extract()
                    if not raw_rows or len(raw_rows) < 2:
                        continue

                    df = parse_table(raw_rows)
                    if df is None or df.empty:
                        continue

                    # Skip fragment tables
                    if len(df.columns) <= 1 and len(df) <= 1:
                        continue

                    headers = list(df.columns)
                    data_rows = df.values.tolist()

                    # Get table bounding box for text context
                    bbox = table_obj.bbox  # (x0, top, x1, bottom)
                    tbl_top = bbox[1]
                    tbl_bottom = bbox[3]

                    # Caption detection: look for "Table X" text near table
                    caption = ""
                    for w in words:
                        # Check above the table (within 60px)
                        if tbl_top - 60 < w['bottom'] < tbl_top:
                            if w['text'].lower() == 'table':
                                caption_line_top = w['top']
                                caption_words = [
                                    (cw['x0'], cw['text']) for cw in words
                                    if abs(cw['top'] - caption_line_top) < 5
                                ]
                                caption_words.sort(key=lambda x: x[0])
                                caption = ' '.join(cw[1] for cw in caption_words)
                                break

                    # Text before table (within 200px above)
                    text_before_words = [
                        w for w in words
                        if tbl_top - 200 < w['bottom'] < tbl_top - 10
                    ]
                    text_before = ' '.join(
                        w['text'] for w in sorted(text_before_words,
                                                   key=lambda w: (w['top'], w['x0']))
                    )

                    # Text after table (within 260px below)
                    text_after_words = [
                        w for w in words
                        if tbl_bottom + 10 < w['top'] < tbl_bottom + 260
                    ]
                    text_after = ' '.join(
                        w['text'] for w in sorted(text_after_words,
                                                   key=lambda w: (w['top'], w['x0']))
                    )

                    table_id = f"{pdf_name}_table_{actual_page_num}_{tbl_idx}"

                    extracted.append({
                        "table_id": table_id,
                        "page": actual_page_num,
                        "table_index": tbl_idx,
                        "section_number": current_section,
                        "section_title": current_section_title,
                        "caption": caption,
                        "headers": headers,
                        "rows": data_rows,
                        "text_before": text_before,
                        "text_after": text_after,
                    })

                except Exception as e:
                    logger.error(f"  Could not extract table on page {actual_page_num}: {e}")

    if temp_pdf_path and os.path.exists(temp_pdf_path):
        os.unlink(temp_pdf_path)

    if target_section is not None:
        logger.info(f"\nExtracted {len(extracted)} tables matching 'Table {target_section}*'")
    else:
        section_tables = len([t for t in extracted if t.get('section_number')])
        logger.info(f"\nExtracted {len(extracted)} tables ({section_tables} with section info)")
    return extracted


def extract_copper_grades(tables, min_grade=0.01, max_grade=3.0):
    """
    Scan extracted tables for copper grade values using pandas.

    Searches both column headers AND row labels for Cu/Copper patterns.
    Extracts numeric values that fall within the expected grade range.

    Strategy:
      1. Column match: header contains cu/copper → extract values from that column
      2. Row match: row label contains cu/copper → extract numeric values from that row
      3. Bonus: "head grade" / "assay" columns used only when table has copper context

    Args:
        tables: list of table dicts from extract_tables()
        min_grade: minimum valid Cu % (default 0.01)
        max_grade: maximum valid Cu % (default 3.0)

    Returns: list of dicts:
        [{value, page, table_id, column_header, row_context, section_number}, ...]
    """
    import re
    import pandas as pd

    cu_pattern = re.compile(r'\bcu\b|\bcopper\b|\bcu\s*[\(%]', re.IGNORECASE)
    bonus_pattern = re.compile(r'\bhead\s*grade\b|\bassay\b', re.IGNORECASE)

    grades = []

    for table in tables:
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        if not rows:
            continue

        meta = {
            "page": table.get("page"),
            "table_id": table.get("table_id"),
            "section_number": table.get("section_number"),
        }

        col_names = headers[:len(rows[0])] if headers else None
        if col_names:
            seen = {}
            for j, h in enumerate(col_names):
                if h in seen:
                    seen[h] += 1
                    col_names[j] = f"{h}_{seen[h]}"
                else:
                    seen[h] = 1
        df = pd.DataFrame(rows, columns=col_names)

        # --- Strategy 1: columns with Cu/Copper in header ---
        cu_cols = [c for c in df.columns if cu_pattern.search(str(c))]

        # --- Strategy 3 (bonus): "Head Grade"/"Assay" with copper context ---
        if not cu_cols:
            context = " ".join([
                str(table.get("text_before", "")),
                str(table.get("text_after", "")),
                str(table.get("caption", "")),
            ])
            if cu_pattern.search(context):
                cu_cols = [c for c in df.columns if bonus_pattern.search(str(c))]

        if cu_cols:
            # Extract first number from each cell, coerce to float
            row_labels = df.iloc[:, 0].astype(str).str.strip()
            for col in cu_cols:
                numeric = (
                    df[col].astype(str).str.strip()
                    .str.extract(r'(\d+\.?\d*)', expand=False)
                )
                numeric = pd.to_numeric(numeric, errors="coerce")
                mask = numeric.between(min_grade, max_grade)
                for idx in mask[mask].index:
                    grades.append({
                        "value": numeric[idx],
                        "column_header": str(col).strip(),
                        "row_context": row_labels[idx],
                        **meta,
                    })

        # --- Strategy 2: rows with Cu/Copper in label ---
        elif len(df.columns) > 1:
            row_labels = df.iloc[:, 0].astype(str).str.strip()
            cu_rows = row_labels[row_labels.str.contains(
                cu_pattern, na=False
            )]
            if not cu_rows.empty:
                data_cols = df.columns[1:]
                for idx in cu_rows.index:
                    for col in data_cols:
                        cell = str(df.at[idx, col]).strip()
                        m = re.search(r'(\d+\.?\d*)', cell)
                        if not m:
                            continue
                        value = float(m.group(1))
                        if min_grade <= value <= max_grade:
                            grades.append({
                                "value": value,
                                "column_header": str(col).strip(),
                                "row_context": cu_rows[idx],
                                **meta,
                            })

    # Deduplicate: same value from the same row of the same table is one grade
    seen = set()
    unique_grades = []
    for g in grades:
        key = (g["value"], g.get("row_context", ""), g.get("page"), g.get("table_id"))
        if key not in seen:
            seen.add(key)
            unique_grades.append(g)

    logger.info(f"Found {len(unique_grades)} unique copper grade values "
          f"(from {len(grades)} raw matches) in range {min_grade}-{max_grade}%")
    return unique_grades

# Table extraction + copper grade evaluation
if __name__ == "__main__":
    import time

    logger.info("\n=== Table Extraction & Copper Grade Search ===")
    folder = Path(__file__).parent / "test_pdfs"
    pdf_files = sorted(f for f in folder.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDFs in {folder}\n")

    total_start = time.time()
    all_grades = []
    for pdf_path in pdf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"PDF: {pdf_path.name}")
        logger.info(f"{'='*60}")

        try:
            pdf_start = time.time()
            tables = extract_tables(pdf_location=str(pdf_path), target_section="13")
            extract_time = time.time() - pdf_start
            logger.info(f"  Section 13 tables: {len(tables)}  ({extract_time:.1f}s)")

            if tables:
                grade_start = time.time()
                grades = extract_copper_grades(tables)
                grade_time = time.time() - grade_start
                for g in grades:
                    logger.info(f"    Cu {g['value']:.3f}% | col: {g['column_header']} "
                          f"| row: {g['row_context']} | page {g['page']}")
                all_grades.append({"pdf": pdf_path.name, "grades": grades,
                                   "time": extract_time})
            else:
                logger.info("  No Table 13* found")
                all_grades.append({"pdf": pdf_path.name, "grades": [],
                                   "time": extract_time})

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_grades.append({"pdf": pdf_path.name, "grades": [], "error": str(e)})

    total_time = time.time() - total_start

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY  (total: {total_time:.1f}s)")
    logger.info(f"{'='*60}")
    for entry in all_grades:
        n = len(entry["grades"])
        t = entry.get("time", 0)
        if entry.get("error"):
            logger.error(f"  {entry['pdf']}: ERROR — {entry['error']}")
        elif n == 0:
            logger.info(f"  {entry['pdf']}: no Cu grades found  ({t:.1f}s)")
        else:
            values = [g["value"] for g in entry["grades"]]
            logger.info(f"  {entry['pdf']}: {n} Cu grade(s) — "
                  f"min={min(values):.3f}% max={max(values):.3f}% "
                  f"avg={sum(values)/n:.3f}%  ({t:.1f}s)")


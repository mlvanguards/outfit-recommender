import base64
from io import BytesIO

from PIL import Image


def pil_image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded string."""
    byte_io = BytesIO()
    pil_image.save(byte_io, format="JPEG")
    byte_image = byte_io.getvalue()
    base64_image = base64.b64encode(byte_image).decode("utf-8")
    return base64_image

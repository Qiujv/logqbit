"""SVG to ICO converter using PySide6.

This script converts SVG files to multi-resolution ICO files without requiring
external dependencies beyond PySide6 (which is already a dependency of logqbit).

The ICO file will contain 7 resolutions: 16x16, 24x24, 32x32, 48x48, 64x64, 128x128, 256x256.

Usage:
    # Convert default browser.svg to browser.ico
    python svg2ico.py
    
    # Convert any SVG to ICO
    python svg2ico.py input.svg output.ico
"""

import io
import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

# ICO format supports these standard sizes
SIZES = (16, 24, 32, 48, 64, 128, 256)

def pix_to_png_bytes(pix: QPixmap) -> bytes:
    """Convert QPixmap to PNG bytes.
    
    Args:
        pix: QPixmap to convert
        
    Returns:
        PNG image data as bytes
    """
    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    buffer.open(QIODevice.WriteOnly)
    pix.save(buffer, "PNG")
    buffer.close()
    return bytes(byte_array)

def write_ico(png_map: dict[int, bytes], outfile: str):
    """Write multiple PNG images as a single ICO file.
    
    ICO format structure:
    - ICONDIR header (6 bytes): reserved(2) + type(2) + count(2)
    - ICONDIRENTRY for each image (16 bytes each)
    - PNG image data
    
    Reference: https://en.wikipedia.org/wiki/ICO_(file_format)
    
    Args:
        png_map: Dictionary mapping size to PNG bytes
        outfile: Output ICO file path
    """
    # 1. 先收集目录项
    entries = []
    offset = 6 + len(png_map) * 16   # 文件头 + 目录长度
    for size in sorted(png_map):
        png = png_map[size]
        # 在 ICO 格式中，256 用 0 表示
        w = 0 if size == 256 else size
        h = 0 if size == 256 else size
        # 宽/高/颜色数/保留 各 1 字节, 色彩平面/位深度 各 2 字节, 数据大小/偏移 各 4 字节
        entries.append(struct.pack("<BBBBHHLL",
                                   w, h, 0, 0, 1, 32,
                                   len(png), offset))
        offset += len(png)

    # 2. 写文件头
    with open(outfile, "wb") as ico:
        ico.write(struct.pack("<HHH", 0, 1, len(png_map)))  # 保留+类型+数量
        ico.write(b"".join(entries))                         # 目录
        # 3. 写图像数据
        for size in sorted(png_map):
            ico.write(png_map[size])

def svg_to_ico(svg_path: str, ico_path: str):
    """Convert SVG file to multi-resolution ICO file.
    
    Creates an ICO file containing the following sizes:
    16x16, 24x24, 32x32, 48x48, 64x64, 128x128, 256x256
    
    Args:
        svg_path: Path to input SVG file
        ico_path: Path to output ICO file
        
    Raises:
        ValueError: If SVG file is invalid
    """
    # 确保 QApplication 存在
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # 加载 SVG
    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        raise ValueError(f"Invalid SVG file: {svg_path}")
    
    png_map = {}
    for s in SIZES:
        # 创建透明背景的 pixmap
        pix = QPixmap(s, s)
        pix.fill(Qt.transparent)
        
        # 使用 QPainter 渲染 SVG
        painter = QPainter(pix)
        renderer.render(painter, QRectF(0, 0, s, s))
        painter.end()
        
        png_map[s] = pix_to_png_bytes(pix)
    
    write_ico(png_map, ico_path)
    print(f"✓ Created ICO file: {ico_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # 命令行参数模式
        svg_to_ico(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        # 默认模式：转换项目中的 browser.svg
        svg_path = Path(__file__).parent.parent / "assets" / "browser.svg"
        ico_path = Path.cwd() / "browser.ico"
        
        if not svg_path.exists():
            print(f"Error: SVG file not found: {svg_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Converting {svg_path.name} to {ico_path.name}...")
        svg_to_ico(str(svg_path), str(ico_path))
    else:
        print("Usage:")
        print("  python svg2ico.py input.svg output.ico")
        print("  python svg2ico.py  (converts browser.svg to browser.ico)")
        sys.exit(1)

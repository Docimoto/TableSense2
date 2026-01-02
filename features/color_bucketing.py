"""
Color bucketing for Excel cell colors.

Converts RGB colors from openpyxl into a fixed set of 8 semantic color buckets
using HLS (Hue, Lightness, Saturation) color space classification.

Per Appendix A of the requirements document.
"""

from typing import Optional, Tuple
import colorsys
from openpyxl.styles.colors import COLOR_INDEX


# Color bucket indices
COLOR_BUCKETS = {
    'black': 0,
    'white': 1,
    'gray': 2,
    'red': 3,
    'yellow': 4,
    'green': 5,
    'blue': 6,
    'other': 7,
}

# Thresholds for color classification
S_GRAY_MAX = 0.15  # Maximum saturation for achromatic (gray/black/white)
L_WHITE_MIN = 0.95  # Minimum lightness for white
L_BLACK_MAX = 0.2  # Maximum lightness for black

# Standard Excel theme colors (RGB values)
# These are default theme colors used when workbook theme is not available
# Theme indices: 0=dark1, 1=light1, 2=dark2, 3=light2, 4=accent1, 5=accent2, etc.
# Note: Dark1 (theme 0) varies by theme - using a medium gray that works well with tints
# When tint=-0.5 is applied: (128, 128, 128) * 0.5 = (64, 64, 64) which classifies as gray
THEME_COLORS = {
    0: (128, 128, 128),  # dark1 (medium gray - works for both light and dark gray shades)
    1: (255, 255, 255),  # light1 (white)
    2: (68, 68, 68),     # dark2 (dark gray)
    3: (91, 155, 213),   # light2 (light blue - adjusted to match common Excel themes where light2 can be blue-tinted)
    4: (68, 114, 196),   # accent1 (blue)
    5: (237, 125, 49),   # accent2 (orange)
    6: (165, 165, 165),  # accent3 (gray)
    7: (255, 192, 0),    # accent4 (yellow)
    8: (91, 155, 213),   # accent5 (light blue)
    9: (112, 173, 71),   # accent6 (green)
    10: (255, 0, 0),     # hyperlink (red)
    11: (0, 176, 80),    # followedHyperlink (green)
}


class ColorBucketer:
    """
    Converts RGB colors to semantic color buckets.
    
    Uses HLS color space to classify colors into 8 buckets:
    black, white, gray, red, yellow, green, blue, other
    """
    
    @staticmethod
    def color_to_rgb_tuple(color) -> Optional[Tuple[int, int, int]]:
        """
        Resolve an openpyxl.styles.colors.Color into (r, g, b) with 0-255 ints.
        
        Args:
            color: openpyxl Color object (or None)
            
        Returns:
            Tuple of (r, g, b) with values 0-255, or None if color cannot be resolved
        """
        if color is None:
            return None
        
        # Direct ARGB (e.g., "FF00FF00")
        if getattr(color, "type", None) == "rgb" and color.rgb:
            argb = color.rgb
            if len(argb) == 8:  # ARGB
                # Check alpha channel - if fully transparent (00), return None
                # so it can be handled as default (white for backgrounds)
                alpha = argb[0:2]
                try:
                    alpha_val = int(alpha, 16)
                    if alpha_val == 0:
                        # Fully transparent - return None to use default
                        return None
                except ValueError:
                    pass
                hexval = argb[2:]
            else:
                hexval = argb[-6:]
            try:
                r = int(hexval[0:2], 16)
                g = int(hexval[2:4], 16)
                b = int(hexval[4:6], 16)
                return (r, g, b)
            except (ValueError, IndexError):
                return None
        
        # Indexed palette color
        if getattr(color, "type", None) == "indexed" and color.indexed is not None:
            try:
                if isinstance(COLOR_INDEX, tuple) and 0 <= color.indexed < len(COLOR_INDEX):
                    argb = COLOR_INDEX[color.indexed]
                elif isinstance(COLOR_INDEX, dict):
                    argb = COLOR_INDEX.get(color.indexed)
                else:
                    return None
                
                if argb:
                    hexval = argb[-6:]
                    try:
                        r = int(hexval[0:2], 16)
                        g = int(hexval[2:4], 16)
                        b = int(hexval[4:6], 16)
                        return (r, g, b)
                    except (ValueError, IndexError):
                        return None
            except (IndexError, TypeError):
                return None
        
        # Theme color with tint
        if getattr(color, "type", None) == "theme" and hasattr(color, "theme"):
            theme_idx = color.theme
            tint = getattr(color, "tint", 0.0)
            
            # Special handling for theme 1 (light1) - in Excel themes, light1 is typically
            # black for text (default font color) and white for backgrounds
            # When used as default font color (theme=1, tint=0), it should be black
            # We'll handle this in color_to_bucket by checking the context (font vs background)
            # For now, use the standard mapping but note that theme 1 may need special handling
            
            # Get base theme color RGB
            base_rgb = THEME_COLORS.get(theme_idx)
            if base_rgb is None:
                # Unknown theme index, default to gray
                base_rgb = (128, 128, 128)
            
            # Special handling for theme 1 (light1/white) with positive tint
            # When white is tinted in Excel, it produces light gray, not pure white
            # Excel's behavior: tinting white (255,255,255) with positive tint darkens it to gray
            if theme_idx == 1 and tint > 0:
                # Use a light gray base that, when lightened, produces the expected light gray
                # Formula: if we want final lightness around 0.95, start from ~0.85
                # For RGB, lightness 0.85 ≈ RGB(217, 217, 217)
                base_rgb = (217, 217, 217)  # Light gray base for tinted white
            
            # Apply tint
            # Excel tint formula:
            # - For positive tints (lightening): new = old + (255 - old) * tint
            # - For negative tints (darkening): new = old * (1 + tint)
            r, g, b = base_rgb
            if tint > 0:
                r = int(r + (255 - r) * tint)
                g = int(g + (255 - g) * tint)
                b = int(b + (255 - b) * tint)
            elif tint < 0:
                r = int(r * (1 + tint))
                g = int(g * (1 + tint))
                b = int(b * (1 + tint))
            
            # Clamp to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            return (r, g, b)
        
        return None
    
    @staticmethod
    def rgb_to_bucket(r: int, g: int, b: int) -> int:
        """
        Map RGB color to one of 8 color buckets using HLS classification.
        
        Args:
            r: Red channel (0-255)
            g: Green channel (0-255)
            b: Blue channel (0-255)
            
        Returns:
            Bucket index (0-7): black=0, white=1, gray=2, red=3, yellow=4, green=5, blue=6, other=7
        """
        # Normalize to [0, 1] for colorsys
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # Convert RGB to HLS
        h, lightness, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        
        # Achromatic check (black/white/gray)
        if s <= S_GRAY_MAX:
            if lightness >= L_WHITE_MIN:
                return COLOR_BUCKETS['white']
            elif lightness <= L_BLACK_MAX:
                return COLOR_BUCKETS['black']
            else:
                return COLOR_BUCKETS['gray']
        
        # Chromatic mapping by hue ranges
        # Hue ranges are in [0, 1] (fraction of 360°)
        if s > S_GRAY_MAX:
            # Red: h < 0.0944 (0-34°) or h >= 0.8333 (300-360°)
            # Expanded to include red-orange (up to 34°) and pink/magenta (300-360°)
            if h < 0.0944 or h >= 0.8333:
                return COLOR_BUCKETS['red']
            # Yellow: 0.0944 <= h < 0.2222 (34-80°)
            elif 0.0944 <= h < 0.2222:
                return COLOR_BUCKETS['yellow']
            # Green: 0.2222 <= h < 0.4722 (80-170°)
            elif 0.2222 <= h < 0.4722:
                return COLOR_BUCKETS['green']
            # Blue: 0.4722 <= h < 0.8056 (170-290°)
            # This includes cyan (180° = 0.5) and blue-green shades
            elif 0.4722 <= h < 0.8056:
                return COLOR_BUCKETS['blue']
            # Other: any chromatic color not matching above (purples between 290-300°)
            # Note: Most purples/pinks are now captured in red range (h >= 0.8333)
            else:
                return COLOR_BUCKETS['other']
        
        # Fallback (should not reach here)
        return COLOR_BUCKETS['other']
    
    @classmethod
    def color_to_bucket(cls, color, default_for_none: str = 'black') -> int:
        """
        Convert an openpyxl Color object directly to a bucket index.
        
        Args:
            color: openpyxl Color object (or None)
            default_for_none: Bucket name to use when color is None or when theme=1 (light1) 
                             is used as default font color.
                             'black' for default font color, 'white' for default background.
                             Defaults to 'black'.
            
        Returns:
            Bucket index (0-7). When color is None:
            - For font colors: defaults to 'black' (0) - Excel's default font color
            - For background colors: should use 'white' (1) - Excel's default background
            - Otherwise defaults to the specified default_for_none bucket
            
            Special handling: theme=1 (light1) with tint=0.0 is typically black for fonts
            and white for backgrounds. When default_for_none='black', treat theme=1 as black.
        """
        # Handle None color
        if color is None:
            return COLOR_BUCKETS.get(default_for_none, COLOR_BUCKETS['other'])
        
        # Special handling: theme=1 (light1) with tint=0.0 is Excel's default font color
        # In most Excel themes, light1 for text is black, not white
        # When default_for_none='black' (font color), treat theme=1 as black
        if (getattr(color, "type", None) == "theme" and 
            hasattr(color, "theme") and 
            color.theme == 1 and 
            getattr(color, "tint", 0.0) == 0.0 and
            default_for_none == 'black'):
            # Theme 1 (light1) used as default font color should be black
            return COLOR_BUCKETS['black']
        
        rgb = cls.color_to_rgb_tuple(color)
        if rgb is None:
            # When color cannot be resolved, use the specified default
            return COLOR_BUCKETS.get(default_for_none, COLOR_BUCKETS['other'])
        return cls.rgb_to_bucket(*rgb)


# openpyxl excel workbook cell features

Extract celll specific features from openpyxl.



| Feature | Values | Description |
| :---- | :---- | :---- |
| data type | s, n, b, f, e, d | Treat "s", "inlineStr", "str" as “text-like”. Treat "n" as numeric (plus a special case for truly empty cells). "b" – Boolean: A True/False value. Example: True, False. "e" – Error. Contains an error code: \#DIV/0\!, \#REF\!, \#N/A, etc. "d" \- This is data\_type \== "d" or data\_type \== "n" with date format. Convert into 6-channel one-hot |
| Font color | black, white, gray, red, yellow, green, blue, other | used for both font and background. We will put it into 8 fixed buckets. Convert into one hot 8-channel one-hot. See Appendix A |
| Background color | black, white, gray, red, yellow, green, blue, other | used for both font and background. We will put it into 8 fixed buckets. Convert into one hot 8-channel one-hot.  See Appendix A  |
| digit proportion | 0 to 1 | digit proportion. non-digit proportion is inverse |
| length of data | s,m,l,xl | One-hot 4-channel Bucket. See Appendix B |
| font size | s,m,l | One-hot 3-channel Bucket. See Appendix C |
| font into small medium large |
| has bold | 0 or 1 |  |
| has strikethrough | 0 or 1 |  |
| strikethrough proportion | 0 to 1 | This is a percentage |
| border | left, right, top, bottom | can have one to 4 of these |
| is locked | 0 or 1 |  |
| is hidden | 0 or 1 |  |
| is merged left | 0 or 1 |  |
| Is merged right | 0 or 1 |  |
| Is merged top | 0 or 1 |  |
| Is merged bottom | 0 or 1 |  |
| is in table | 0 or 1 |  |
| is in table header | 0 or 1 |  |
| Has format | 0 or 1 | Can be any kind of format |
| Has comment | 0 or 1 |  |



# Appendix A - Cell Color Bucketing Specification 

## 1. Objective

Normalize arbitrary 24-bit RGB colors from Excel (via `openpyxl`) into a **small, stable set of semantic color buckets** suitable for feature engineering in docAI’s Excel table models.

We care about:

- Rough color semantics (black / white / gray / red / yellow / green / blue).
- Distinguishing “normal” vs “highlight/status” cells.
- Being robust to slightly different shades used by different carriers.

## 2. Buckets

We define the following **8 color buckets**:

- `black`
- `white`
- `gray`
- `red`
- `yellow`
- `green`
- `blue`
- `other`  (anything that doesn’t clearly fit the above)

Implementation detail:

- The classifier is defined over **all** 24-bit RGB colors.
- Every RGB maps into one of these buckets.
- `"other"` is used to keep the main buckets “clean” and to catch unusual hues (teal, purple, etc.) or edge cases.

Additionally, callers may use a separate sentinel (e.g. `"none"` or `None`) when a color cannot be resolved at all from Excel theme/index information. That is outside this spec; this spec covers **RGB → bucket**.

## 3. Inputs and assumptions

- Source: `openpyxl.styles.colors.Color` objects from:
  - `cell.fill.fgColor` (background)
  - `cell.font.color` (font)
- Color may be specified as:
  - Direct ARGB hex (`type == "rgb"`, e.g. `FF00FF00`).
  - Indexed palette (`type == "indexed"`).
  - Theme + tint (`type == "theme"`).  
    - v0.1: theme+tint support is optional; unresolved colors may be treated as “unknown” by caller.

- We operate on standard 8-bit channels: `0 ≤ r,g,b ≤ 255`.

## 4. Classification algorithm (concept)

1. **Resolve RGB**
   - Convert the `Color` object to `(r, g, b)` (0–255).
   - If resolution fails (theme/tint not handled yet), return `None` and let the caller decide (usually “default” Excel colors).

2. **Convert to HLS**
   - Use Python `colorsys.rgb_to_hls` to get:
     - `h` (hue ∈ [0,1])
     - `l` (lightness ∈ [0,1])
     - `s` (saturation ∈ [0,1])

3. **Achromatic check (black/white/gray)**
   - If saturation is low (`s ≤ S_GRAY_MAX`), treat as achromatic:
     - If `l ≥ L_WHITE_MIN` → `white`
     - Else if `l ≤ L_BLACK_MAX` → `black`
     - Else → `gray`

   Initial thresholds (tunable):
   - `S_GRAY_MAX = 0.15`
   - `L_WHITE_MIN = 0.92`
   - `L_BLACK_MAX = 0.15`

4. **Chromatic mapping (red/yellow/green/blue/other)**

   For `s > S_GRAY_MAX`, we classify by **hue ranges**. Hue ranges are in [0,1] (fraction of 360°):

   - `red`:
     - `h < 0.0556` (0–20°) or `h ≥ 0.9444` (340–360°)
   - `yellow`:
     - `0.0556 ≤ h < 0.25` (20–90°)
   - `green`:
     - `0.25 ≤ h < 0.4722` (90–170°)
   - `blue`:
     - `0.4722 ≤ h < 0.8056` (170–290°)
   - `other`:
     - Any chromatic color not matching the above (e.g. purples/cyans).

These boundaries are **initial defaults** and may be tuned later based on real-world Excel samples.

## 5. Features derived from buckets (non-normative)

Typical feature encoding per cell:

- Background color: one-hot over the 8 buckets (8 channels).
- Font color: one-hot over the 8 buckets (8 channels).
- Optional extra:
  - `font_dark_on_light_bg` (1 channel):
    - e.g. `True` if font in `{black, gray}` and background in `{white, yellow}`.

Total: up to **17 color-related channels** if all are used.

## 6. Python reference implementation

### 6.1 Resolve openpyxl `Color` → RGB

```python
from openpyxl.styles.colors import COLOR_INDEX

def color_to_rgb_tuple(color):
    """
    Resolve an openpyxl.styles.colors.Color into (r, g, b) with 0–255 ints.
    Returns None if the color cannot be resolved (e.g. theme with tint not handled).
    """
    if color is None:
        return None

    # Direct ARGB (e.g. "FF00FF00")
    if getattr(color, "type", None) == "rgb" and color.rgb:
        argb = color.rgb
        if len(argb) == 8:  # ARGB
            hexval = argb[2:]
        else:
            hexval = argb[-6:]
        try:
            r = int(hexval[0:2], 16)
            g = int(hexval[2:4], 16)
            b = int(hexval[4:6], 16)
            return (r, g, b)
        except ValueError:
            return None

    # Indexed palette color
    if getattr(color, "type", None) == "indexed" and color.indexed is not None:
        argb = COLOR_INDEX.get(color.indexed)
        if argb:
            hexval = argb[-6:]
            try:
                r = int(hexval[0:2], 16)
                g = int(hexval[2:4], 16)
                b = int(hexval[4:6], 16)
                return (r, g, b)
            except ValueError:
                return None

    # NOTE: Theme + tint handling can be added later if needed.
    return None

# Appendix B  - Length of Data Buckets

Take length of data and convert into buckets

s < 10 
m < 50 
l < 100 
xl > 99

Convert this into 4-channel one-hot.

# Appendix C - Font size Buckets

Take sz as float, normalize by the sheet median or common size:

size_ratio = cell.font.sz / sheet_median_font_size

Convert these into buckets

0 = s  (sz < 0.8 * median)
1 = m  (0.8 * median ≤ sz ≤ 1.2 * median)
2 = l  (sz > 1.2 * median)

Convert this into 3-channel one-hot.

#

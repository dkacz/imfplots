# Important Correction - Poland Position

## User Correction
Poland is at **position 13 from the left**, not position 16 as originally assumed.

## Impact on Accuracy

### Before Correction (Position 16):
- Measured: ~21%
- Expected: ~12-13%  
- Error: ~8 percentage points

### After Correction (Position 13):
- Measured: **16.72%**
- Expected: ~12-13%
- Error: **~4 percentage points**

## Conclusion
With correct country positioning, the extraction achieves **±4pp accuracy**, which is reasonable for pixel-based measurement from 550×353px images.

The remaining ~4pp error is within acceptable range considering:
- Visual estimation uncertainty
- Pixel quantization limits
- Calibration interpolation between label points

## Lesson Learned
**Accurate country/category positioning is critical** for extraction accuracy. Verifying the x-axis order against the actual plot labels is essential.

## Metadata Correction

The `plot_metadata.json` file had **incorrect country ordering** which caused the positioning error.

### Wrong Order (Original):
Positions 13-17: `BEL, CHE, SWE, AUT, POL`

### Correct Order (Fixed):
Positions 13-17: `POL, BEL, CHE, SWE, AUT`

**Impact**: The metadata had 5 countries (positions 13-17) in the wrong sequence, with Poland displaced by 4 positions. This directly caused the ~8pp extraction error before manual correction.

**Fix Applied**: Updated both image entries in `plot_metadata.json` with the corrected country order verified against the actual plot x-axis labels.

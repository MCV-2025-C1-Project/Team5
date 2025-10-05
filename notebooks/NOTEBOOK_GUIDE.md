# testrun.ipynb - Quick Guide

## Overview
The `testrun.ipynb` notebook now supports custom descriptor and distance selection to speed up experimentation.

## How to Use

### 1. Open the Notebook
```bash
jupyter notebook testrun.ipynb
```

### 2. Configure Your Test (Last Cell)

Find the configuration section in the last cell:

```python
# ============================================================================
# SELECT DESCRIPTORS AND DISTANCES (Comment/Uncomment to customize)
# ============================================================================

# Option 1: Evaluate ALL combinations (50 total) - Takes ~25 minutes
DESCRIPTORS = None  # None means all
DISTANCES = None    # None means all

# Option 2: Quick test with single combination (1 total) - Takes ~30 seconds
# DESCRIPTORS = ['hsv']
# DISTANCES = ['chi_2.compute_chi_2_distance']
```

### 3. Choose a Preset or Customize

**Simply uncomment the option you want to use!**

## Available Presets

### Option 1: ALL Combinations (Default)
```python
DESCRIPTORS = None
DISTANCES = None
```
- **Combinations:** 50
- **Time:** ~25 minutes
- **Use when:** You want comprehensive results

### Option 2: Quick Test
```python
DESCRIPTORS = ['hsv']
DISTANCES = ['chi_2.compute_chi_2_distance']
```
- **Combinations:** 1
- **Time:** ~30 seconds
- **Use when:** Testing if everything works

### Option 3: Subset Test
```python
DESCRIPTORS = ['rgb', 'hsv', 'lab']
DISTANCES = ['chi_2.compute_chi_2_distance', 'euclidean_distance',
             'hellinger_kernel', 'l1.compute_l1_distance']
```
- **Combinations:** 12
- **Time:** ~6 minutes
- **Use when:** Testing a specific subset

### Option 4: Compare Color Spaces
```python
DESCRIPTORS = ['rgb', 'hsv', 'ycbcr', 'lab', 'grayscale']
DISTANCES = ['chi_2.compute_chi_2_distance']
```
- **Combinations:** 5
- **Time:** ~2.5 minutes
- **Use when:** You want to see which color space works best

### Option 5: Compare Distances
```python
DESCRIPTORS = ['hsv']
DISTANCES = None  # All distances
```
- **Combinations:** 10
- **Time:** ~5 minutes
- **Use when:** You want to see which distance metric works best

## Custom Configuration

You can create your own combination:

```python
# Your custom selection
DESCRIPTORS = ['hsv', 'lab']
DISTANCES = ['chi_2.compute_chi_2_distance', 'hellinger_kernel']
```

### Available Descriptors (5 total)
- `'rgb'` - RGB color histogram
- `'hsv'` - HSV color histogram
- `'ycbcr'` - YCbCr color histogram
- `'lab'` - LAB color histogram
- `'grayscale'` - Grayscale histogram

### Available Distances (10 total)
- `'euclidean_distance'` - Euclidean (L2) distance
- `'l1.compute_l1_distance'` - L1 (Manhattan) distance
- `'chi_2.compute_chi_2_distance'` - Chi-Square distance
- `'histogram_intersection.compute_histogram_intersection'` - Histogram Intersection
- `'hellinger_kernel'` - Hellinger distance
- `'cosine.compute_cosine_similarity'` - Cosine distance
- `'canberra.canberra_distance'` - Canberra distance
- `'bhattacharyya.bhattacharyya_distance'` - Bhattacharyya distance
- `'jensen_shannon.jeffrey_divergence'` - Jensen-Shannon divergence
- `'correlation.correlation_distance'` - Correlation distance

## Step-by-Step Example

### Quick Test Example:

1. **Edit the last cell** - Uncomment Option 2:
```python
# Option 2: Quick test with single combination (1 total) - Takes ~30 seconds
DESCRIPTORS = ['hsv']
DISTANCES = ['chi_2.compute_chi_2_distance']
```

2. **Comment out Option 1** (add `#` at the start):
```python
# Option 1: Evaluate ALL combinations (50 total) - Takes ~25 minutes
# DESCRIPTORS = None
# DISTANCES = None
```

3. **Run all cells** (Cell → Run All)

4. **Check output** - You'll see:
```
Total combinations to evaluate: 1
Estimated time: ~0.5 minutes
```

## Tips

1. **Start small:** Always test with 1-2 combinations first
2. **Progressive testing:** Expand gradually based on results
3. **Time estimates:** Actual time may vary based on your hardware
4. **Values per bin:** Adjust `VALUES_PER_BIN` for different histogram resolutions

## Troubleshooting

### Error: "Invalid descriptor"
- Check spelling: must be exactly `'rgb'`, `'hsv'`, etc. (lowercase)
- Use single quotes: `'hsv'` not `"hsv"`

### Error: "Invalid distance metric"
- Use exact names from the list above
- Include the full name: `'chi_2.compute_chi_2_distance'`

### Too slow?
- Reduce number of combinations
- Try Option 2 (single combination) first

## Summary

**To run a quick test:**
1. Uncomment Option 2 in the last cell
2. Comment Option 1
3. Run all cells
4. Check results in ~30 seconds

**To run full analysis:**
1. Keep Option 1 uncommented (DESCRIPTORS = None, DISTANCES = None)
2. Run all cells
3. Wait ~25 minutes for complete results

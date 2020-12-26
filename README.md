# Filter discription
## bilateral Filter

Python: cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) → dst¶

Parameters:	
- src – Source 8-bit or floating-point, 1-channel or 3-channel image.
- dst – Destination image of the same size and type as src .
- d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
- sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
- sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .

### Scripts
- bilateral_filter_times
  - bilateral filterの試行回数ごとの比較
- bilateral_filter_ds
  - dごとでの比較
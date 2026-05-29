# ISP pipeline

Stage-by-stage reference for the forward image-signal-processing pipeline:
linear scene radiance in, 8-bit sRGB out. Each stage lists its equation, the
units of its parameters, and a citation. See
[ADR 004](adr/004-isp-ordering.md) for why the stages run in this order.

> TODO: fill in equations + citations per stage as each lands (Phase 2,
> slice 9). Skeleton below mirrors the implemented stage order.

## 1. Vignetting (`lens.py`)
## 2. Distortion (`lens.py`)
## 3. Bayer mosaic (`bayer.py`)
## 4. Integration / exposure (`noise.py`)
## 5. Dark current (`noise.py`)
## 6. Shot noise (`noise.py`)
## 7. Read noise (`noise.py`)
## 8. Black level + normalize (`color.py`)
## 9. AWB gains (`color.py`)
## 10. Demosaic, bilinear (`demosaic.py`)
## 11. Color correction matrix (`color.py`)
## 12. Gamma (`color.py`)
## 13. 8-bit quantize (`color.py`)

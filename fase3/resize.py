import openslide
import numpy as np
from PIL import Image

input_path = "008e5c_0.tif"
output_path = "008e5c_0_tissue2048.jpg"
crop_size = 2048  # tamaño del parche final (puedes bajarlo a 1024 si quieres)

slide = openslide.OpenSlide(input_path)

# 1) Trabajamos en el nivel más bajo (thumbnail grande pero manejable)
level_thumb = slide.level_count - 1
tw, th = slide.level_dimensions[level_thumb]
print(f"Nivel thumbnail {level_thumb} con tamaño {tw}x{th}")

thumb = slide.read_region((0, 0), level_thumb, (tw, th)).convert("RGB")
thumb_np = np.array(thumb)

# 2) Dividimos en una cuadrícula (ej. 8x8)
grid = 8
tile_w = tw // grid
tile_h = th // grid

best_score = 1e9
best_center = (tw // 2, th // 2)  # fallback: centro
for i in range(grid):
    for j in range(grid):
        x0, x1 = j * tile_w, (j + 1) * tile_w
        y0, y1 = i * tile_h, (i + 1) * tile_h
        tile = thumb_np[y0:y1, x0:x1]

        # promedio de intensidad en escala de grises
        gray = tile.mean(axis=2)
        score = gray.mean()

        # buscamos el tile más oscuro (más tejido, menos blanco)
        if score < best_score:
            best_score = score
            best_center = ( (x0 + x1) // 2, (y0 + y1) // 2 )

print("Mejor celda (más tejido) en thumbnail:", best_center, "score:", best_score)

# 3) Pasamos la coordenada del thumbnail al nivel 0 (max resolución)
w0, h0 = slide.level_dimensions[0]
scale_x = w0 / tw
scale_y = h0 / th

cx0 = int(best_center[0] * scale_x)
cy0 = int(best_center[1] * scale_y)

# 4) Definimos el rectángulo de crop en nivel 0
x = max(0, cx0 - crop_size // 2)
y = max(0, cy0 - crop_size // 2)

# Ajuste por si nos vamos fuera del borde
if x + crop_size > w0:
    x = w0 - crop_size
if y + crop_size > h0:
    y = h0 - crop_size

x = max(0, x)
y = max(0, y)

print(f"Crop en nivel 0: x={x}, y={y}, size={crop_size}x{crop_size}")

region = slide.read_region((x, y), 0, (crop_size, crop_size)).convert("RGB")
region.save(output_path, format="JPEG", quality=90)

slide.close()
print("Listo, exportado parche con tejido como:", output_path)

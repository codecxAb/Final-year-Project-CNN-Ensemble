from PIL import Image, ImageFilter
import numpy as np

img = Image.open("/Users/anurag/College/lungCanerProject/lungcare_triage/frontend-streamlit/assets/lung_texture.png").convert("RGBA")
print("Size:", img.size)

arr = np.array(img)
alpha = arr[:,:,3]
print("Alpha mean:", alpha.mean())
print("Alpha shape:", alpha.shape)

import matplotlib.pyplot as plt
from PIL import Image

# draw a histogram for grey image
def histogram_for_gray_image(image_path: str):
    img = Image.open(image_path)
    gray_img = img.convert('L')
    data = list(gray_img.getdata())

    plt.hist(data, bins=256, range=(0, 255), color='black', alpha=0.7)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(axis='y')
    plt.show() # show histogram

# export functions
__all__ = [
    "histogram_for_gray_image"
]
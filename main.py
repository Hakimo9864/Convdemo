import cv2
import numpy as np
import matplotlib.pyplot as plt

# make two more Kernels
def show_image(title, image):
    plt.figure(figsize=(4, 4))
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def apply_kernel(image, kernel):
    return cv2.filter2D(image, -1, kernel)


img = cv2.imread("data/kitty.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

show_image("Original Grayscale", gray)

kernels = {
    "Blur": np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]),
    "Horizontal Edges": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]),
    "Vertical Edges": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
     #Kernel 1: Sharpen
    "Sharpen": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    #Kernel 2: Edge Detection
    "Edge Detection": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]),

}

for name, kernel in kernels.items():
    result = apply_kernel(gray, kernel)
    show_image(name, result)
    

# Recorded Results
#Kernel name 1: Sharpen

#Kernel values: 0, -1. 0, -1, 5, -1, 0, -1, 0.

#What did the output image show? 

#The sharpen kernel made the image clearer and more detailed. 
# The edges of the cat and other objects became more defined, and the overall image looked sharper with stronger contrast.

#________________________________________________________________________________________________________________________________________

#Kernel name: Edge Detection

#Kernel values: -1, -1, -1, -1, 8, -1, -1, -1, -1

#What did the output image show?
#The edge detection kernel highlighted the edges in the image. 
# Most smooth areas became darker while the outlines of the cat and other shapes appeared bright, showing the boundaries of objects.

#________________________________________________________________________________________________________________________________________





#9. Reflection

#1. Which kernel produced the most interesting result?

#The edge detection kernel produced the most interesting result because it highlighted the outlines of the cat and removed most of the shading, making the structure of the image easier to see.

#________________________________________________________________________________________________________________________________________

#2. Which kernel might help detect an animal in an image?

#The edge detection kernel would help detect an animal in an image because it highlights the boundaries and shapes of objects, which helps identify the outline of the animal.

#________________________________________________________________________________________________________________________________________


#3. Why might detecting edges help a computer recognize objects?

#Detecting edges helps a computer recognize objects because edges define the shapes and boundaries of things in an image. By identifying these outlines, a computer can distinguish one object from another.

#________________________________________________________________________________________________________________________________________

#4. Why is convolution better suited for images than flattening pixels?

#Convolution is better suited for images because it analyzes small groups of neighboring pixels and preserves spatial relationships in the image. Flattening pixels turns the image into a long list of numbers and loses information about how pixels are arranged.
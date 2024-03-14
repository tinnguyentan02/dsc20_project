"""
DSC 20 Project
Name(s): TODO
PID(s):  TODO
Sources: None
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """

        if not isinstance(pixels, list) or not pixels:
            raise TypeError()

        if not all(isinstance(pixel, list) and len(pixel) > 0 for pixel in pixels):
            raise TypeError()
        
        
        num_rows = len(pixels)
        num_cols = len(pixels[0])

        if not all(len(pixel) == num_cols for pixel in pixels):
            raise TypeError()
        
        if not all (isinstance(element, list) and len(element) == 3 for pixel in pixels for element in pixel):
            raise TypeError()
        
        if not all (isinstance(num, int) and 0 <= num <= 255 for pixel in pixels for element in pixel for num in element):
            raise TypeError()
        
        self.pixels = pixels
        self.num_rows = num_rows
        self.num_cols = num_cols

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[value for value in pixel] for pixel in row] for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(list(self.pixels))

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        
        if not 0 <= row <= self.num_rows - 1 or not 0 <= col <= self.num_cols - 1:
            raise ValueError()
        
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        
        if not 0 <= row <= self.num_rows - 1 or not 0 <= col <= self.num_cols - 1:
            raise ValueError()
        
        if not isinstance(new_color, tuple) or len(new_color) != 3 or not all(isinstance(value, int) for value in new_color):
            raise TypeError()
        
        if any(value > 255 for value in new_color):
            raise ValueError()
        
        if any(value < 0 for value in new_color):
            pass
        
        self.pixels[row][col] = [new_color[i] if new_color[i] >= 0 else self.pixels[row][col][i] for i in range(3)]



class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # negated_pixels = [[[255 - component for component in pixel] for pixel in row] for row in image.pixels]
        negated_pixels = list(map(lambda row: list(map(lambda pixel: list(map(lambda component: 255 - component, pixel)), row)), image.pixels))
        # Return a new RGBImage instance with the negated pixels
        return RGBImage(negated_pixels)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        gray_pixels = list(map(lambda row: list(map(lambda pixel: [sum(pixel) // len(pixel)] * 3, row)), image.pixels))

        return RGBImage(gray_pixels)
    

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        rotated_pixels = list(map(lambda row: list(map(lambda pixel: pixel, row[::-1])), image.pixels[::-1]))

        return RGBImage(rotated_pixels)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        
        number_of_pixels = len(image.pixels) * len(image.pixels[0])

        return sum(map(lambda row: sum(list(map(lambda pixel: sum(pixel) // 3, row))), image.pixels)) // number_of_pixels

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if not isinstance(intensity, int):
            raise TypeError()

        if intensity > 255 or intensity < -255:
            raise ValueError()
        
        adjusted_pixels = list(map(lambda row: list(map(lambda pixel: list(map(lambda value: value + intensity if 0 <= value + intensity <= 255 else 255 if value + intensity > 255 else 0, pixel)), row)), image.pixels))

        return RGBImage(adjusted_pixels)
    
    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        
        num_rows, num_cols = len(image.pixels), len(image.pixels[0])

        blurred_pixels = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        for i in range(num_rows):
            for j in range(num_cols):
  
                neighbors = []
                for ni in range(max(0, i - 1), min(i + 2, num_rows)):
                    for nj in range(max(0, j - 1), min(j + 2, num_cols)):
                        neighbors.append(image.pixels[ni][nj])
                
                avg_r = sum(pixel[0] for pixel in neighbors) // len(neighbors)
                avg_g = sum(pixel[1] for pixel in neighbors) // len(neighbors)
                avg_b = sum(pixel[2] for pixel in neighbors) // len(neighbors)

                blurred_pixels[i][j] = [avg_r, avg_g, avg_b]

        return RGBImage(blurred_pixels)
                    
                


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.free_calls = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #

        if self.free_calls == 0:
            self.cost += 5

        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #

        if self.free_calls == 0:
            self.cost += 6

        else:
            self.free_calls -= 1
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        if self.free_calls == 0:
            self.cost += 10
        else:
            self.free_calls -= 1

        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        if self.free_calls == 0:
            self.cost += 1
        else:
            self.free_calls -= 1

        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        if self.free_calls == 0:
            self.cost += 5
        else:
            self.free_calls -= 1

        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """

        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        
        self.free_calls += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        # YOUR CODE GOES HERE #
        
        if not any([0 <= var <= 255 and isinstance(var, int) for row in chroma_image.pixels for pixel in row for var in pixel]):
            raise TypeError()
        
        if not any([0 <= var <= 255 and isinstance(var, int) for row in background_image.pixels for pixel in row for var in pixel]):
            raise TypeError()
        
        for i in range(len(chroma_image.pixels)):
            for j in range(len(chroma_image.pixels[i])):
                if chroma_image.pixels[i][j] == list(color):
                    chroma_image.pixels[i][j] = background_image.pixels[i][j]
                
        return chroma_image

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """

        if not any([0 <= var <= 255 and isinstance(var, int) for row in sticker_image.pixels for pixel in row for var in pixel]):
            raise TypeError()
        
        if not any([0 <= var <= 255 and isinstance(var, int) for row in background_image.pixels for pixel in row for var in pixel]):
            raise TypeError()
        
        if sticker_image.num_rows > background_image.num_rows or sticker_image.num_cols > background_image.num_cols:
            raise ValueError()
            
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        
        if sticker_image.num_rows + y_pos > background_image.num_rows or sticker_image.num_cols + x_pos > background_image.num_cols:
            raise ValueError()
        

        for i in range(len(sticker_image.pixels)):
            for j in range(len(sticker_image.pixels[i])):
                background_image.pixels[i +  x_pos][j + y_pos] = sticker_image.pixels[i][j]

        return background_image
    
    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        
        def to_grayscale(pixel):
            return sum(pixel) // 3

        num_rows, num_cols = len(image.pixels), len(image.pixels[0])
        new_pixels = [[[0, 0, 0] for _ in range(num_cols)] for _ in range(num_rows)]
        threshold = 15  # Edge detection threshold

        for i in range(1, num_rows - 1):
            for j in range(1, num_cols - 1):
                center_intensity = to_grayscale(image.pixels[i][j])
                is_edge = False

                for ni in range(i - 1, i + 2):
                    for nj in range(j - 1, j + 2):
                        if ni == i and nj == j:
                            continue  # Skip the center pixel itself

                        neighbor_intensity = to_grayscale(image.pixels[ni][nj])
                        if abs(center_intensity - neighbor_intensity) > threshold:
                            is_edge = True
                            break  # Break the inner loop

                    if is_edge:
                        break  # Break the outer loop

                new_pixels[i][j] = [255, 255, 255] if is_edge else [0, 0, 0]

        # Create a new image object with `new_pixels` as its pixel data
        new_image = Image(num_rows, num_cols, new_pixels)
        return new_image
                            
                            


# # Part 5: Image KNN Classifier #
# class ImageKNNClassifier:
#     """
#     Represents a simple KNNClassifier
#     """

#     def __init__(self, k_neighbors):
#         """
#         Creates a new KNN classifier object
#         """
#         # YOUR CODE GOES HERE #

#     def fit(self, data):
#         """
#         Stores the given set of data and labels for later
#         """
#         # YOUR CODE GOES HERE #

#     def distance(self, image1, image2):
#         """
#         Returns the distance between the given images

#         >>> img1 = img_read_helper('img/steve.png')
#         >>> img2 = img_read_helper('img/knn_test_img.png')
#         >>> knn = ImageKNNClassifier(3)
#         >>> knn.distance(img1, img2)
#         15946.312896716909
#         """
#         # YOUR CODE GOES HERE #

#     def vote(self, candidates):
#         """
#         Returns the most frequent label in the given list

#         >>> knn = ImageKNNClassifier(3)
#         >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
#         'label2'
#         """
#         # YOUR CODE GOES HERE #

#     def predict(self, image):
#         """
#         Predicts the label of the given image using the labels of
#         the K closest neighbors to this image

#         The test for this method is located in the knn_tests method below
#         """
#         # YOUR CODE GOES HERE #


# def knn_tests(test_img_path):
#     """
#     Function to run knn tests

#     >>> knn_tests('img/knn_test_img.png')
#     'nighttime'
#     """
#     # Read all of the sub-folder names in the knn_data folder
#     # These will be treated as labels
#     path = 'knn_data'
#     data = []
#     for label in os.listdir(path):
#         label_path = os.path.join(path, label)
#         # Ignore non-folder items
#         if not os.path.isdir(label_path):
#             continue
#         # Read in each image in the sub-folder
#         for img_file in os.listdir(label_path):
#             train_img_path = os.path.join(label_path, img_file)
#             img = img_read_helper(train_img_path)
#             # Add the image object and the label to the dataset
#             data.append((img, label))

#     # Create a KNN-classifier using the dataset
#     knn = ImageKNNClassifier(5)

#     # Train the classifier by providing the dataset
#     knn.fit(data)

#     # Create an RGBImage object of the tested image
#     test_img = img_read_helper(test_img_path)

#     # Return the KNN's prediction
#     predicted_label = knn.predict(test_img)
#     return predicted_label

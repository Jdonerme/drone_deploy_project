import cv2
import zbar
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Camera specifications for an Iphone 6
fx=1229
cx=360
fy=1153
cy=640

# The length of the pattern
QR_LENGTH = 8.8 # cm

""" Given camera specifications, calculate the camera matrix.

    Args:
        fx: focal length x component
        cx: principal point x component
        fy: focal length y component
        cy: principal point y component
    returns:
        3 by 3 camera matrix

    """
def calculate_camera_matrix(fx, cx, fy, cy):
    cam_matrix = np.zeros((3, 3))
    cam_matrix[0][0] = fx
    cam_matrix[0][2] = cx
    cam_matrix[1][1] = fy
    cam_matrix[1][2] = cy
    cam_matrix[2][2] = 1

    return cam_matrix

""" 
    Locates the corners of a QR code in an image.

    Locates the corners using zbar. The order of the corners will be the same
    for the same QR code even if the image is rotated. If the QR code can not
    be detected, None is returned.

    Args: 
        img: the name of the image to search with
    returns:
        locs: 2D numpy array containing the 4 corners

    """
def get_corner_locs(img):
    try:
        # obtain image data
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY,dstCn=0)
        pil = Image.fromarray(gray)
        width, height = pil.size
        raw = pil.tobytes()

        # wrap image data
        image = zbar.Image(width, height, 'Y800', raw)

        # scan the image for barcodes
        scanner.scan(image)

        locs = []
        # extract results
        for symbol in image:
            locs.append(np.array(symbol.location, dtype='float32'))
        return np.array(locs)[0]
    except:
        return None


"""  Displays a visualzation of camera positon based on the rotation
     vector and translation vector

    Args: 
        img: the name of the image
        rvec: rotation vector of the camera
        tvec: translation vector of the camera
    

        """
def generate_visualization(img, rotvec, tvec):
    # If our Z coordinate will be negative, we need to reverse the vector
    # because all photos are taken from above.
    if rotvec[2] < 0:
        rvec = -1 * rotvec
    else:
        rvec = rotvec

    xs = [rvec[0] * tvec[0]]
    ys = [rvec[1] * tvec[1]]
    zs = [rvec[2] * tvec[2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs=zs, label="camera")
    ax.scatter([0],[0], label="Image")

    l = 5 # to scale the arrows
    ax.quiver(xs[0], ys[0], zs[0], \
        -rvec[0] * l, - rvec[1] * l,  - rvec[2] * l, arrow_length_ratio=0.1)
    
    ax.set_xlabel('x distance (cm)')
    ax.set_ylabel('y distance (cm)')
    ax.set_zlabel(' Height (cm)')
    plt.legend(loc='best')
    title = "Camera location in " + img
    plt.title(title)
    plt.show()


""" This program locates the QR code in all images, and compares their
    orientation with the orientation of the pattern to detect camera 
    location and visualize it.

    """

if __name__ == "__main__" :
    images = ["IMG_67"] * 9
    for i in range(9):
        images[i] += str(i + 19) +".jpg"

    cam_matrix = calculate_camera_matrix(fx, cx, fy, cy)
    # create a reader
    scanner = zbar.ImageScanner()

    # configure the reader
    scanner.parse_config('enable')

    # Find the corners of the pattern image
    scale_corners = get_corner_locs("pattern.jpg")

    # shift the corners so that in 3D space we consider our squares centered at
    # 0, 0
    length = scale_corners[1][1] - scale_corners[1][0]
    scale_corners -= scale_corners[0]
    scale_corners -= length / 2
    scale_corners /= (length / QR_LENGTH)

    # make 3D space 3 Dimensional by assuming Z component is 0
    objp = np.zeros((4,3))
    objp[:,:-1] = scale_corners
    #objp =scale_corners

    for img in images:
        # get the location of the corners
        locs = get_corner_locs(img)
        if locs is None:
            print "QR code could not be located in img " + img

        else:
            success, rvec, tvec = cv2.solvePnP(objp, locs, cam_matrix, None)
            if success:
                generate_visualization(img, rvec, tvec)   
            else:
                print "No visualization able to be generated for img: " + img
        
      
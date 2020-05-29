import re
import string
from PIL import Image, ImageOps
from brand_replacements import brand_map
import cv2
import math
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition

def resize(original_image, target_size=(500,500), bg_color=(255, 255, 255)): 
    """resize image by custom height and width"""
    
    original_image.thumbnail(target_size, Image.ANTIALIAS)
    resized_image = Image.new("RGB", target_size, bg_color) # blank canvas
    
    # paste imge into canvas
    resized_image.paste(original_image, (int((target_size[0] - original_image.size[0]) / 2),\
                                         int((target_size[1] - original_image.size[1]) / 2))) 
    return resized_image


def process_name(old_name, folder):
    """preprocess image names"""
    
    ext = re.search(r'(.jpg|.png)',old_name).group() # get file extention
    new_name = old_name.lower()
    new_name = re.search(r'(\\[\w\s.()\'"-%-@&*$+,‘’]+.png|\\[\w\s.()\'"-%-$@&*+,‘’]+.jpg)',new_name).group()
    new_name = re.sub(ext,'',new_name) 
    
    new_name = re.sub(r'\s','_',new_name) # replace whitespce with underscore
    new_name = re.sub(r'^[\d]+','',new_name) # remove leading digits
    new_name = re.sub(r'(\"|\')','',new_name) # remove quotations
    
    # replace punctuations and leading digits
    punctuations = [punct for punct in string.punctuation if punct != '_']+['‘’']
    punctuations = ''.join(punctuations)
    new_name = new_name.translate(str.maketrans('', '', punctuations)) # take out puncturations
    if folder != 'non-hype':
        new_name = re.sub(r'^[0-9]+','',new_name) 
    
    return f'{new_name}{ext}' # join folder into name and file extention


def get_brand(name):
    """helper to get brand of the shoe"""
    
    brand = re.search(r'^_?[a-zA-Z]+', name).group()
    # corner cases
    if brand == 'air':
        if 'jordan' in name:
            return 'jordan'
        return 'nike'
    
    if brand == 'ace':
        if name == 'ace_16_purecontrol_ultra_boost_kith_flamingos':
            return 'adidas'
        return 'gucci'
    
    if brand == 'undefeated':
        if 'air' in name:
            return 'nike'
        return 'adidas'
    
    if brand == 'comme':
        if 'air' in name:
            return 'nike'
        return 'converse'
    
    if brand == 'fear':
        if 'air' in name:
            return 'nike'
        if 'chuck' in name:
            return 'converse'
        return 'fear_fo_god'
    
    if brand =='offwhite':
        if 'jordan' in name:
            return 'jordan'
        if 'zoom_fly' in name:
            return 'nike'
        return 'offwhite'
    
    # if brand is already correct
    if brand not in brand_map:
        return brand
    
    # newly mapped brands
    return brand_map[brand]


def get_info(image, name, folder, path):
    """get basic info of a shoe image including brand and dimensions"""
    
    name = re.sub(r'(.png|.jpg)', '', name)
    if folder == 'non-hype':
        return [name, image.size[0], image.size[1], folder, path]
    brand = get_brand(name)
    return [name, brand, image.size[0], image.size[1], folder, path]


def calc_angles(lines):
    """calulate all arctan(y/x)"""
    angles = []   
    for line_dim in lines:
        x1, y1, x2, y2 = line_dim[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    return angles


def detect_angle_orient(edges,facing=False,rho=1):
    """Detect if the image is flat, outputs the rotated angle
       Also can detect the facing of shoe (set higher rho)"""
    
    lines = cv2.HoughLinesP(edges, rho, math.pi/180.0, 100, minLineLength=0.01, maxLineGap=0.001)
    
    if lines is None:
        return False
    
    angles = calc_angles(lines)
    median_angle = np.median(angles)
    if not facing:
        return median_angle
    negative_angles = sum(1 for angle in angles if angle < 0)
    positive_angles = sum(1 for angle in angles if angle > 0)
    if negative_angles > positive_angles:
        return True
    return False


def fix_orientation(file_name):
    """Detect orientation of an image. 
        If rotated, make horizontal. 
        If orientated to the right, mirror the image"""
    
    original_img = cv2.imread(file_name)
    
    # grayscale, detect edges and detect houghlines
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 100, apertureSize=3)
    rotate_angle = detect_angle_orient(edges)
    
    # rotate to new orientation
    img_rotated = ndimage.rotate(original_img, rotate_angle, cval=255)
    
    # if shoe is pointing left, point right (need to set rho to higher value)
    if detect_angle_orient(edges,facing=True,rho=5):
        img_rotated = cv2.flip(img_rotated, 1)
        
    file_name = re.search(r'(\\[\w\s.()\'"-%-@&*$+,‘’]+.png|\\[\w\s.()\'"-%-$@&*+,‘’]+.jpg)',file_name).group()
    file_name = re.sub(r'\\','',file_name)
    return img_rotated, file_name
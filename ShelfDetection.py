import random

import cv2
import numpy as np
import math
import argparse
import glob


# Perform edge detection
def hough_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    # cv2.imwrite('../pictures/output/opening.jpg', opening)
    #cv2.imshow("opening",opening)
    edges = cv2.Canny(opening, 50, 150, apertureSize=3)  # Canny edge detection
    #cv2.imshow("edge_image",edges)
    #cv2.waitKey(0)
    # cv2.imwrite('../pictures/output/canny.jpg', edges)
    threshold = 250
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    while(lines==None):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # Hough line detection
        threshold -= 1
    while(len(lines[0]) < 20):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # Hough line detection
        threshold -= 1
        #print(lines)
        #print(threshold)
    hough_lines = []

    # Lines are represented by rho, theta; convert to endpoint notation
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            hough_lines.append(((x1, y1), (x2, y2)))

    #cv2.imwrite('hough.jpg', img)
    return hough_lines

# Random sampling of lines
def sample_lines(lines, size):
    if size > len(lines):
        size = len(lines)
    return random.sample(lines, size)


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Lines don't cross

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y


# Find intersections between multiple lines (not line segments!)
def find_intersections(lines, img):
    intersections = []
    for i in xrange(len(lines)):
        line1 = lines[i]
        for j in xrange(i + 1, len(lines)):
            line2 = lines[j]

            if not line1 == line2:
                intersection = line_intersection(line1, line2)
                if intersection:  # If lines cross, then add
                    # Don't include intersections that happen off-image
                    # Seems to cost more time than it saves
                    # if not (intersection[0] < 0 or intersection[0] > img.shape[1] or
                    #                 intersection[1] < 0 or intersection[1] > img.shape[0]):
                    # print 'adding', intersection[0],intersection[1],img.shape[1],img.shape[0]
                    intersections.append(intersection)

    return intersections

#length of a vector
def length_vector(vec):
    return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

#find angle between two lines (x1,y1) and (x2,y2) is line1 ,(x3,y3) and (x4,y4) is line2.
def angle(x1,y1,x2,y2,x3,y3,x4,y4):
    line1 = [x2-x1,y2-y1]
    line2 = [x4-x3,y4-y3]
    angle = math.asin(det(line1, line2)/(length_vector(line1)* length_vector(line2)));
    return angle

# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img, intersections):
    # Image dimensions
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Grid dimensions
    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = None

    for i in xrange(grid_rows):
        for j in xrange(grid_columns):
            cell_left = i * grid_size
            cell_right = (i + 1) * grid_size
            cell_bottom = j * grid_size
            cell_top = (j + 1) * grid_size
            cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

            current_intersections = 0  # Number of intersections in the current cell
            for x, y in intersections:
                if cell_left < x < cell_right and cell_bottom < y < cell_top:
                    current_intersections += 1

            # Current cell has more intersections that previous cell (better)
            if current_intersections > max_intersections:
                max_intersections = current_intersections
                best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)

    print(best_cell)
    if not best_cell == [None, None]:
        rx1 = best_cell[0] - grid_size / 2
        ry1 = best_cell[1] - grid_size / 2
        rx2 = best_cell[0] + grid_size / 2
        ry2 = best_cell[1] + grid_size / 2
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 10)
        cv2.imwrite('center.jpg', img)

    return best_cell
    
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input dataset of images")
args = vars(ap.parse_args())

# loop over the images
for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# load the image, convert it to grayscale, and blur it slightly
	img = cv2.imread(imagePath)    
        #img = cv2.imread("/home/sneha/Academics/sem8/MP/Mini_Project1/images/10.jpg")
        lines = hough_transform(img)
        #print(len(lines))
        l= sample_lines(lines, len(lines))
        intersections = find_intersections(l,img)
        x_avg = 0;
        y_avg = 0;
        count = 0;
        for x in intersections:
            #print x[0],x[1]
            if(x[0] < 20000) and (x[0] > -20000) and (x[1] < 20000) and (x[1] > -20000):
                count+=1
                x_avg +=x[0]
                y_avg +=x[1]
        #print count
        #print x_avg/count
        #print y_avg/count
        vanishing_point = [x_avg/count, y_avg/count]
        top_right = [img.shape[1],0]
        bottom_right = [img.shape[1],img.shape[0]]
        required_angle = angle(vanishing_point[0], vanishing_point[1], bottom_right[0], bottom_right[1], vanishing_point[0], vanishing_point[1], top_right[0], top_right[1])
        #print (required_angle * 180)/math.acos(-1)
        step = img.shape[0] / 100
        #print "Following are the intersections:"
        result_lines = []
        for x in range(0,99):
            line1 = [vanishing_point, [img.shape[1],img.shape[0] - x*step]]
            line2 = [top_right, bottom_right]
            (x1,y1) = line_intersection(line1, line2)
            #print x1 , y1
            line1 = [vanishing_point, [ img.shape[1] , img.shape[0] - x*step]]
            line2 = [[0 , 0], [0 , img.shape[0]]]
            (x2 , y2) = line_intersection(line1, line2)
            #print x2, y2
            #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            result_lines.append(((x1, y1), (x2, y2)))
        for x in result_lines:
            for y in lines:
                if (abs(angle(x[0][0],x[0][1],x[1][0],x[1][1],y[0][0],y[0][1],y[1][0],y[1][1])) < 0.0005):
                    cv2.line(img, x[0], x[1], (0, 0, 255), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)
            #best_cell = find_vanishing_point(img, 2, intersections)

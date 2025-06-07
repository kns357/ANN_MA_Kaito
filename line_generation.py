
import numpy as np
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import random as ra
import math

def neighbors_8(img, y, x):
    h, w = img.shape
    for ny in range(max(y-1,0), min(y+2,h)):
        for nx in range(max(x-1,0), min(x+2,w)):
            if (ny, nx) != (y, x):
                yield ny, nx

def find_primary_continuous_path(skeleton_image):
    if not np.any(skeleton_image):
        return []

    active_pixels = set(zip(*np.where(skeleton_image)))
    if not active_pixels:
        return []

    start_point = None
    for r, c in active_pixels:
        count = sum(1 for nr, nc in neighbors_8(skeleton_image, r, c) if skeleton_image[nr, nc])
        if count <= 1:
            start_point = (r, c)
            break
    if not start_point:
        start_point = next(iter(active_pixels))
        
    path = []
    visited = set()
    stack = [start_point]

    while stack:
        current_pixel = stack.pop()
        
        if current_pixel in visited:
            continue
        
        visited.add(current_pixel)
        path.append(current_pixel)

        unvisited_neighbors = []
        for ny, nx in neighbors_8(skeleton_image, *current_pixel):
            if skeleton_image[ny, nx] and (ny, nx) not in visited:
                unvisited_neighbors.append((ny, nx))
        
        for neighbor in unvisited_neighbors:
            stack.append(neighbor)

    return path

def bresenham_line(y0, x0, y1, x1):
    points = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def segment_only_on_white_pixels(start, end, skeleton):
    points_on_line = bresenham_line(start[0], start[1], end[0], end[1])
    return all(skeleton[y, x] != 0 for y, x in points_on_line)

def line_approximation_with_corners(path, skeleton, angle_tolerance_degrees):
    segments = []
    if len(path) < 2:
        return segments

    current_segment_start_point = path[0]

    for i in range(1, len(path)):
        current_point_on_path = path[i]

        if i == len(path) - 1:
            if segment_only_on_white_pixels(current_segment_start_point, current_point_on_path, skeleton):
                segments.append((current_segment_start_point, current_point_on_path))
            break

        next_point_on_path = path[i + 1]

        v1 = np.array(current_point_on_path) - np.array(current_segment_start_point)
        v2 = np.array(next_point_on_path) - np.array(current_point_on_path)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue

        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        cos_theta = max(-1.0, min(1.0, dot_product / (magnitude_v1 * magnitude_v2)))

        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)

        is_corner = (angle_degrees > angle_tolerance_degrees and
                     angle_degrees < (180 - angle_tolerance_degrees))

        if is_corner or not segment_only_on_white_pixels(current_segment_start_point, current_point_on_path, skeleton):
            if segment_only_on_white_pixels(current_segment_start_point, current_point_on_path, skeleton):
                segments.append((current_segment_start_point, current_point_on_path))
            current_segment_start_point = current_point_on_path

    if current_segment_start_point != path[-1] or not segments:
        if len(path) > 1 and (not segments or segments[-1][1] != path[-1]):
            if segment_only_on_white_pixels(current_segment_start_point, path[-1], skeleton):
                segments.append((current_segment_start_point, path[-1]))

    return segments
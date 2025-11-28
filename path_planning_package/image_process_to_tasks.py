import json
import sys
import os
import cv2
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import yaml
from python_motion_planning import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../path_planning_package/")

# Load the JSON configuration
with open("path_planning_package/config.json") as f:
    config = json.load(f)

configuration = config["Configuration"]
robot = config["Robot"]

# Load map filepath and parameters from configuration
map_filepath = os.path.join("path_planning_package",configuration["map_filepath"])

rr = 1  # robot["robot_radius"]
params = {}
params["MAX_V"] = robot["maximum_linear_velocity"]
params["MAX_W"] = robot["maximum_angular_velocity"]

# Determine map type based on the algorithm

api_key = configuration['api_key']
genai.configure(api_key=api_key)

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def detect_colored_dots(image_path, threshold=20):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, green, yellow, and cyan dots in HSV
    color_ranges = {
        'red': ((0, 100, 100), (10, 255, 255)),
        'green': ((40, 50, 50), (80, 255, 255)),
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'cyan': ((80, 100, 100), (100, 255, 255))
    }

    # Dictionary to store detected points
    detected_points = {'red': [], 'green': [], 'yellow': [], 'cyan': []}

    # Loop over color ranges and detect points
    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the specified color range
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Find contours for the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get the center of the contour using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])  # X coordinate of the dot
                cY = int(M["m01"] / M["m00"])  # Y coordinate of the dot
                
                # Check if the new point is within the threshold distance of any existing point
                too_close = False
                for existing_point in detected_points[color]:
                    if euclidean_distance((cX, cY), existing_point) < threshold:
                        too_close = True
                        break
                
                # Only add the point if it's not too close to an existing point
                if not too_close:
                    # Draw a circle around the dot on the original image
                    cv2.circle(image, (cX, cY), 1, (0, 0, 0), -1)
                    
                    # Add the coordinates as text next to the detected point
                    coordinates_text = f"({cX},{cY})"
                    cv2.putText(image, coordinates_text, (cX-10, cY+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                    # Store the detected point by color
                    detected_points[color].append((cX, cY))

    return detected_points, image

def extract_roi(image, center, size=20):
    """Extract a region of interest (ROI) from the image around the given center point."""
    x, y = center
    roi = image[max(y-size, 0):min(y+size, image.shape[0]), max(x-size, 0):min(x+size, image.shape[1])]
    return roi

detected_points, processed_image = detect_colored_dots(map_filepath)

# Print detected points in the original image size
print("Detected points:")
for key, points in detected_points.items():
    print(f"{key} points: {points}")

# Optionally, save or display the processed image with the marked dots
# cv2.imshow("Detected Dots", processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Optionally save the image
cv2.imwrite('output_image_with_dots.png', processed_image)

# Now process the ROIs for each detected point
results = []
# results = [{'color': 'red', 'coordinates': (71, 51), 'text': '1 \n'}, {'color': 'green', 'coordinates': (319, 421), 'text': '1p \n'}, {'color': 'green', 'coordinates': (155, 423), 'text': '1q'}, {'color': 'green', 'coordinates': (199, 386), 'text': '1b \n'}, {'color': 'green', 'coordinates': (299, 382), 'text': '1c \n'}, {'color': 'green', 'coordinates': (355, 382), 'text': '1G/1n \n'}, {'color': 'green', 'coordinates': (379, 346), 'text': '1h/1m \n'}, {'color': 'green', 'coordinates': (332, 349), 'text': '1d/1f \n'}, {'color': 'green', 'coordinates': (380, 315), 'text': '1k \n'}, {'color': 'green', 'coordinates': (333, 316), 'text': '1E \n'}, {'color': 'green', 'coordinates': (115, 107), 'text': '1R \n'}, {'color': 'green', 'coordinates': (154, 49), 'text': '1a \n'}, {'color': 'yellow', 'coordinates': (74, 107), 'text': '1'}]
import time
for color, points in detected_points.items():
    if color == 'cyan':
        continue
    for point in points:
        roi = extract_roi(cv2.imread(map_filepath), point)
        
        # Save the ROI temporarily
        roi_path = "temp_roi.png"
        cv2.imwrite(roi_path, roi)
        
        myfile = genai.upload_file(roi_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([myfile, "\n\n", f"Can you provide the blue text which is the label for {color} colored dot in the dead center? If there is a red or blue dot in the center, it will be a number or multiple numbers separated by forward slash. If it is a green dot in the center, it will be a number followed by a letter, or many of these types of code (number followed by letter) separated by forward slash. Answer just the text and nothing else."])
        
        # Append the color, coordinates, and recognized text to results
        results.append({
            'color': color,
            'coordinates': point,
            'text': result.text
        })
        print(results)
        # time.sleep(10)

# Dictionary to organize tasks
tasks = {}

# Process each item
for item in results:
    task_numbers = item['text'].strip().split('/')  # Get task numbers from text
    for task_number in task_numbers:
        task_number = task_number.strip()  # Clean up the task number
        
        # For start points
        if item['color'] == 'red':
            for tn in task_number.split('/'):
                tn = tn.strip()
                if tn not in tasks:
                    tasks[tn] = {'start': item['coordinates'], 'intermediate': [], 'end': None}
        
        # For intermediate points
        elif item['color'] == 'green':
            for tn in task_number.split('/'):
                tn = tn.strip()
                task_number = tn[0]
                if task_number in tasks:
                    tasks[task_number]['intermediate'].append((item['coordinates'], tn))

        # For end points
        elif item['color'] == 'yellow':
            for tn in task_number.split('/'):
                tn = tn.strip()
                if tn in tasks:
                    tasks[tn]['end'] = item['coordinates']

# Creating task sequences
task_sequences = []

# Build the final task sequences
for task_number, details in tasks.items():
    task_sequence = [details['start']]  # Start point
    # Sort intermediate points by the letter following the task number
    sorted_intermediate = sorted(details['intermediate'], key=lambda x: (x[1].lower(), x[0]))
    task_sequence.extend([coords for coords, _ in sorted_intermediate])  # Intermediate points sorted
    if details['end']:
        task_sequence.append(details['end'])  # End point
    
    task_sequences.append((task_number, task_sequence))  # Store with task number

image_loader = ImageProcessor(rr=rr)
image_loader.load_image(map_filepath)
env, image, scale_factor = image_loader.process()
image = image.astype(np.uint8)
image = image.transpose()[::-1, :]
height, width = image.shape

black_locations = np.argwhere(image == 0)
obstacles = [[int(coord[1]), image.shape[0] - int(coord[0])] for coord in black_locations]
dimensions = [image.shape[1], image.shape[0]]

# Create an RGB image
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
rgb_image[:, :, 0] = image  
rgb_image[:, :, 1] = image  
rgb_image[:, :, 2] = image
# Save processed image
cv2.imwrite('Processed_image.png', rgb_image)
y_adj = cv2.imread(map_filepath).shape[0]

final_tasks = []
# Print the task sequences
for task_number, sequence in task_sequences:
    print(f"Task {task_number}: {[(round(end[0]/scale_factor), round((y_adj - end[1])/scale_factor)) for end in sequence]}")
    final_tasks.append([[round(end[0]/scale_factor), round((y_adj - end[1])/scale_factor)] for end in sequence])

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate the average point between two points
def average_point(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

# Helper function to check if two line segments intersect and return the intersection point
def line_intersection(p1, p2, q1, q2):
    """Returns the intersection point of two line segments if they intersect, else returns None."""
    def det(a, b, c, d):
        return a * d - b * c

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    denom = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
    if denom == 0:
        return None  # Lines are parallel

    intersect_x = det(det(x1, y1, x2, y2), x1 - x2, det(x3, y3, x4, y4), x3 - x4) / denom
    intersect_y = det(det(x1, y1, x2, y2), y1 - y2, det(x3, y3, x4, y4), y3 - y4) / denom

    # Check if the intersection point is on both line segments
    if (
        min(x1, x2) <= intersect_x <= max(x1, x2) and
        min(y1, y2) <= intersect_y <= max(y1, y2) and
        min(x3, x4) <= intersect_x <= max(x3, x4) and
        min(y3, y4) <= intersect_y <= max(y3, y4)
    ):
        return (intersect_x, intersect_y)
    return None

# Funtion to calculate angle between 3 points
def calculate_angle(p1, p2, p3):
    """Calculate the angle between the three points p1, p2, and p3."""
    # Create vectors from points
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    
    # Normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate the dot product and the angle
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)  # Angle in radians
    angle_deg = np.degrees(angle_rad)  # Convert to degrees

    return angle_deg

#Funtion to merge close nodes
def merge_close_nodes(points, angle_threshold=20):
    merged_points = []
    i = 0

    while i < len(points):
        # Check if we have at least three points to form an angle
        if i < len(points) - 2:
            angle = calculate_angle(points[i], points[i + 1], points[i + 2])
            print(angle)
            if abs(angle) < angle_threshold:
                # Merge the three points by keeping the first one
                merged_points.append(points[i])
                i += 3  # Skip the next two points
                continue
        
        # If no merging occurred, keep the current point
        merged_points.append(points[i])
        i += 1

    return merged_points

# Function to check for intersections between paths and add common nodes
def handle_intersections(paths):
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1 = paths[i]
            path2 = paths[j]

            # Compare every segment in path1 with every segment in path2
            for k in range(len(path1) - 1):
                for l in range(len(path2) - 1):
                    intersection_point = line_intersection(path1[k], path1[k + 1], path2[l], path2[l + 1])
                    if intersection_point:
                        # Insert the intersection point into both paths if not already present
                        if intersection_point not in path1:
                            path1.insert(k + 1, intersection_point)
                        if intersection_point not in path2:
                            path2.insert(l + 1, intersection_point)
    return paths

# Funtion to generate waypoints
def process_paths(points, unit_distance=10.0):
    """
    Generate a path where all points are at a specified distance from the previous one,
    with special handling for the endpoint.

    :param points: List of points (tuples or lists) [(x1, y1), (x2, y2), ...].
    :param unit_distance: The distance between consecutive points.
    :return: List of interpolated points with unit distance between them.
    """
    def distance(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def interpolate(p1, p2, dist):
        """
        Interpolate points between p1 and p2 such that they are at a specific distance apart.
        """
        num_points = int(np.ceil(dist / unit_distance))
        dx = (p2[0] - p1[0]) / num_points
        dy = (p2[1] - p1[1]) / num_points
        return [(round(p1[0] + i * dx, 2), round(p1[1] + i * dy, 2)) for i in range(num_points)]

    path = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = distance(p1, p2)
        
        if dist > unit_distance:
            path.extend(interpolate(p1, p2, dist))
        else:
            path.append(p1)
    
    path.append(points[-1])
    
    # Handle the last point and special endpoint case
    if len(path) >= 2:
        last_point = path[-1]
        second_last_point = path[-2]
        remaining_dist = distance(last_point, points[-1])
        
        if remaining_dist < 0.5 * unit_distance:
            # Remove last interpolated point and add the endpoint
            path[-1] = points[-1]
        else:
            # Remove last interpolated point and split the distance
            path.pop()
            mid_point = (
                round((second_last_point[0] + points[-1][0]) / 2, 2),
                round((second_last_point[1] + points[-1][1]) / 2, 2)
            )
            path.append(mid_point)
            path.append(points[-1])
    else:
        # Directly add the last point if the path has less than 2 points
        path.append(points[-1])

    return path

# Funtion to find the closest point
def find_closest_point(point_list, given_point):
    """Find the closest point to a given point from a list of points."""
    if not point_list:
        return None  # Return None if the list is empty

    closest_point = point_list[0]
    min_distance = euclidean_distance(closest_point, given_point)

    for point in point_list[1:]:
        distance = euclidean_distance(point, given_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point

planned_paths = {}
def plan_task(task_points):
    full_path = []
    for i in range(len(task_points) - 1):
        start = task_points[i]
        end = task_points[i + 1]
        # Check both (start, end) and (end, start) in planned_paths
        if (start, end) not in planned_paths and (end, start) not in planned_paths:
            print((round(start[0]/scale_factor), round((y_adj - start[1])/scale_factor)), (round(end[0]/scale_factor), round((y_adj - end[1])/scale_factor)))
            planner = ThetaStar((round(start[0]/scale_factor), round((y_adj - start[1])/scale_factor)), (round(end[0]/scale_factor), round((y_adj - end[1])/scale_factor)), env)
            cost, path, _ = planner.plan()
            planned_paths[(start, end)] = path
        
        # Append the planned path (always use start -> end direction for consistency)
        full_path.append(planned_paths.get((start, end), planned_paths.get((end, start))))
    return full_path

# Function to plan for all tasks
def plan_all_tasks(tasks):
    all_paths = []
    for task_points in tasks:
        task_points += [task_points[0]]
        task_paths = plan_task(task_points)
        all_paths.append(task_paths)
    return all_paths

# Plan paths for all tasks
sorted_task_sequences = sorted(task_sequences, key=lambda x: x[0])
all_task_paths = plan_all_tasks([tasks for i, tasks in sorted_task_sequences])

parking_paths = []

final_paths = []
for key, path in planned_paths.items():
    final_paths.append(path)
seen = set()
final_paths_final = []
for path in final_paths:
    start = path[0]
    end = path[-1]
    if (start,end) in  seen or (end,start) in seen:
        continue
    else:
        final_paths_final.append(process_paths(path))
        seen.add((start, end))
final_paths = final_paths_final

points_list = []
for path in final_paths:
    points_list += path

agents = []
non_task_endpoints = []
for i, cyan_point in enumerate(detected_points['cyan']):
    start = (round(cyan_point[0]/scale_factor), round((y_adj - cyan_point[1])/scale_factor))
    end = find_closest_point(points_list, start)
    agents.append({'start':list(start), 'name':f'agent{i}'})
    non_task_endpoints.append(list(start))
    parking_paths.append(process_paths([start, end]))
 
d = {'agents':agents, 'map':{'dimension': dimensions, 'obstacles': obstacles, 'non_task_endpoints': non_task_endpoints}, 'tasks':final_tasks}

with open('environment.yaml', 'w') as file:
    yaml.dump(d, file, default_flow_style=False)

print(final_paths)
plot = Plot(final_tasks[0][0], final_tasks[0][-1], env)
plot.plotEnv('Theta*')
for path in final_paths:
    plot.plotPath(path)
for path in parking_paths:
    plot.plotPath(path)
plt.show()


print(parking_paths)
with open('paths.pkl', 'wb') as f:
      pickle.dump(final_paths + parking_paths, f)

    


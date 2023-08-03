import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from Polygon import Polygon
from Video import VideoHandler
from config import *

# Defining the zones
zone1, zone2, zone3, zone4 = (
    Polygon(ZONE1_VERTICES),
    Polygon(ZONE2_VERTICES),
    Polygon(ZONE3_VERTICES),
    Polygon(ZONE4_VERTICES),
)
# Initializing the count for all zones
zone1_count, zone2_count, zone3_count, zone4_count = set(), set(), set(), set()

# Intializing count from z1 to z4 and z4 to z1
zone1_to_zone4_count, zone4_to_zone1_count = 0, 0

# Creating a copy to compare changes in previous and current state z1 to z4
zone1_to_zone4_count_copy = set()
# Source video and Output video
source_video = VideoHandler(VIDEO_PATH)
processed_video = f'{VIDEO_PATH.split(".")[0]}-output.mp4'

# Extracting frames from source video
frames_generator = source_video.get_video_frames_generator()

# Initalizing Videowriter object for writing frames
out = cv2.VideoWriter(
    processed_video,
    cv2.VideoWriter_fourcc(*"avc1"),
    source_video.fps,
    source_video.resolution,
)

trail_history = {}  # stores step coordinates for each id
color_map = {}  # stores color for each id

# Loading the desired model
model = YOLO(MODEL_NAME)
model.fuse()

# Iterating though the frames of source video
for frame in tqdm(frames_generator, total=source_video.total_frames):
    results = model.track(
        source=frame,
        show=False,
        tracker="bytetrack.yaml",
        iou=0.5,
        verbose=False,
        classes=0,
        persist=True,
    )
    # If nothing is detected continue
    if results[0].boxes is None or results[0].boxes.id is None:
        continue
    # Getting detection box cordinates and tracking id
    box_coordinates = results[0].boxes.xyxy.cpu().numpy().astype(float)
    tracker_id = results[0].boxes.id.cpu().numpy().astype(int)

    feet_coordinates = {}  # stores id and feet coordinates

    # Calculating coordinates of feet
    for i in range(len(box_coordinates)):
        # feet location = (xmin+xmax)/2, ymax
        foot_x, foot_y = (
            box_coordinates[i][0] + box_coordinates[i][2]
        ) / 2, box_coordinates[i][3]
        feet_coordinates[tracker_id[i]] = (foot_x, foot_y)

    for key, value in feet_coordinates.items():
        track_id, (x, y) = key, value
        # Storing coordinates of each id
        if track_id not in trail_history:
            trail_history[track_id] = []
        trail_history[track_id].append((int(x), int(y)))
        # Setting colors for each id
        if track_id not in color_map:
            color_map[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        # Checking and adding id's to zone counts
        if zone1.is_point_inside(x, y):
            zone1_count.add(track_id)
        elif zone2.is_point_inside(x, y):
            zone2_count.add(track_id)
        elif zone3.is_point_inside(x, y):
            zone3_count.add(track_id)
        elif zone4.is_point_inside(x, y):
            zone4_count.add(track_id)
        else:
            continue

    zone1_and_zone4_count = zone1_count.intersection(zone4_count)
    # Checking if there has been any moment from zone1 to zone4
    if len(zone1_to_zone4_count_copy) != len(zone1_and_zone4_count):
        added_elements = zone1_and_zone4_count - zone1_to_zone4_count_copy
        for added_element in added_elements:
            # Extracting first and last known coordinates of id
            _, first_y = trail_history[added_element][0]
            _, last_y = trail_history[added_element][-1]
            if first_y > last_y:  # Movement from Zone 1 to Zone 4
                zone1_to_zone4_count += 1
            else:  # Movement from Zone 4 to Zone 1
                zone4_to_zone1_count += 1
        zone1_to_zone4_count_copy = zone1_and_zone4_count.copy()

    # Plotting the segmentation masks over the detected people
    annotated_frame = results[0].plot(boxes=False, probs=False)
    # Converting vertices to arryay for polylines argument
    converted_zone1_array = np.array(ZONE1_VERTICES).reshape(-1, 1, 2)
    converted_zone2_array = np.array(ZONE2_VERTICES).reshape(-1, 1, 2)
    converted_zone3_array = np.array(ZONE3_VERTICES).reshape(-1, 1, 2)
    converted_zone4_array = np.array(ZONE4_VERTICES).reshape(-1, 1, 2)
    # Displaying the zone boundaries
    cv2.polylines(
        annotated_frame,
        [converted_zone1_array],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2,
    )
    cv2.polylines(
        annotated_frame,
        [converted_zone2_array],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )
    cv2.polylines(
        annotated_frame,
        [converted_zone3_array],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2,
    )
    cv2.polylines(
        annotated_frame,
        [converted_zone4_array],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )
    # Displaying count for each zone
    cv2.putText(
        annotated_frame,
        f"Zone-1 count: {len(zone1_count)}",
        (204, 494),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Zone-2 count: {len(zone2_count)}",
        (690, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Zone-3 count: {len(zone3_count)}",
        (908, 194),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Zone-4 count: {len(zone4_count)}",
        (494, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 225, 225),
        2,
    )
    # Displaying common Count
    cv2.putText(
        annotated_frame,
        f"Zone-1 and Zone-4 Common Count: {len(zone1_and_zone4_count)}",
        (20, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Zone-1 to Zone-4 transition: {zone1_to_zone4_count}",
        (20, 40),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Zone-4 to Zone-1 transition: {zone4_to_zone1_count}",
        (20, 60),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 0),
        2,
    )
    # Draw trail lines for each tracked person
    for track_id, trail_points in trail_history.items():
        color = color_map[track_id]
        for i in np.arange(20, len(trail_points), 20):
            cv2.arrowedLine(
                annotated_frame,
                trail_points[i - 20],
                trail_points[i],
                color,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

    out.write(annotated_frame)

out.release()

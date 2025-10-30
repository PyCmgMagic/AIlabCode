import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("../video/traffic1.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("../output/heatmap_output_x.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# For object counting with heatmap, you can pass region points.
# region_points = [(20, 400), (1080, 400)]                                      # line points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points

# Initialize heatmap object
heatmap = solutions.Heatmap(
    show=False,  # display the output
    model="../model/yolo11x.pt",  # path to the YOLO11 model file
    colormap=cv2.COLORMAP_HOT,
    device=0,  # colormap of heatmap
    # region=region_points,  # object counting with heatmaps, you can pass region_points
    # classes=[0, 2],  # generate heatmap for specific classes i.e person and car.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = heatmap(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
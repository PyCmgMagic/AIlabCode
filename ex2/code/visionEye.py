import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("../video/crowdFlow.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("../ouput/visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize vision eye object
visioneye = solutions.VisionEye(
    show=True,  # display the output
    model="../model/yolov8m.pt",  # use any model that Ultralytics support, i.e, YOLOv10
    # classes=[0, 2],  # generate visioneye view for specific classes
    vision_point=(700, 600),  # the point, where vision will view objects and draw tracks
    tracker="bytetrack.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = visioneye(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the video file

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
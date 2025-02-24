#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# for camera
# camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2
# display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

# while display.IsStreaming(): # main loop will go here
# 	img = camera.Capture()

# 	if img is None: #capture timeout
# 		continue

# 	detections = net.Detect(img)
#	print(detections)
#	display.Render(img)
#	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

# for one image
img_path = "street.jpeg"
img = jetson.utils.loadImage(img_path)
detections = net.Detect(img)

# Save detection results to text file
with open("detection_results.txt", "w") as file:
    file.write(f"Detection Results for {img_path}\n")
    file.write(f"Total objects detected: {len(detections)}\n")
    file.write("-" * 50 + "\n\n")
    
    for i, detection in enumerate(detections, 1):
        result = f"""Detection #{i}:
Class ID: {detection.ClassID}
Class Name: {net.GetClassDesc(detection.ClassID)}
Confidence: {detection.Confidence:.2f}
Bounding Box:
    Left:   {detection.Left:.2f}
    Top:    {detection.Top:.2f}
    Right:  {detection.Right:.2f}
    Bottom: {detection.Bottom:.2f}
Center: X: {detection.Center[0]:.2f}, Y: {detection.Center[1]:.2f}
Area: {detection.Area:.2f}
"""
        file.write(result + "\n")
        print(result)  # Also print to console

display = jetson.utils.videoOutput("outimg.jpeg") # 'my_video.mp4' for file

display.Render(img)
display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

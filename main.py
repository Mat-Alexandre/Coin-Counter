from ultralytics import YOLO
import cv2
import cvzone
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Predicting by image
img = cv2.imread("images/samples/1Re15C.jpeg")
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Model initialization and inferation
model = YOLO("yolov8s-brl-coin.pt")
prediction = model(resized)  # return a list of Results objects

results = 0
coin_dict = {
  0: 0.1,
  1: 1.,
  2: .25,
  3: .5,
  4: .05
} 

for r in prediction:
  for box in r.boxes:
    # Extracting th bounding box
    x, y, x1, y1 = box.xyxy[0]
    x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
    w, h = x1 - x, y1 - y

    # Model confidence
    conf = box.conf[0]
    # Naming classes
    cls = box.cls[0]
    current_class = model.names[int(cls)]

    results += coin_dict[int(cls)]
    
    # Showing detection annotations
    cvzone.cornerRect(
      resized, 
      bbox=(x, y, w, h), 
      l=5,
      t=2
    )

    img, _ = cvzone.putTextRect(
      resized, 
      text=f"{current_class} {conf:.2%}", 
      pos=(max(0, x), max(35, y)), 
      scale=.65,
      thickness=1,
      font=cv2.FONT_HERSHEY_SIMPLEX,
      offset=2
    )
img, _ = cvzone.putTextRect(
  resized, 
  text=f"Total R$ {results:.3}", 
  pos=(0, 25), 
  scale=.65,
  thickness=1,
  font=cv2.FONT_HERSHEY_SIMPLEX,
  offset=2
)
# Priting results and wating to exit
cv2.imshow("Coin counter", img)
cv2.waitKey(0)
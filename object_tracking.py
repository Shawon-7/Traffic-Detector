import cv2
import math


cap = cv2.VideoCapture("traffic_vid.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
tracked_objects = {}
object_id = 0


def get_centroid(x, y , w ,h):
    return x + w // 2 , y + h // 2

def euclidean_distance(pt1 , pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 +(pt1[1] - pt2[1])**2)


while True:
    ret , frame = cap.read()


    roi = frame[250:460 , 100:720]
    mask = object_detector.apply(roi)
    _ , mask = cv2.threshold(mask , 254 , 255 , cv2.THRESH_BINARY)
    contours , _ = cv2.findContours(mask , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    current_objects = {}


    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            x, y ,w, h =  cv2.boundingRect(cnt)
            centroid = get_centroid(x , y, w, h)


            found = False

            for obj_id , (cx, cy) in tracked_objects.items():
                if euclidean_distance(centroid , (cx, cy)) <50:
                    current_objects[obj_id] = centroid
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(roi, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    found = True
                    break

            if not found :
                object_id+=1
                current_objects[object_id] = centroid
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(roi, f"ID {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            print(object_id)
    tracked_objects = current_objects.copy()


    cv2.imshow("ROI", roi)
    cv2.imshow("Mask" , mask)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()




import cv2

classNames = []
classfile = "coco.names"


#img = cv2.imread("D:/assignments/hooman.jpg")
# cv2.imshow("Image",img)
# cv2.waitKey(0)
cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)



with open(classfile, 'rt') as f:
    classNames = f.read().rstrip("\n").split("\n")


pbtxt_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model_path = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(model_path, pbtxt_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    classIds, confs ,bboxes = net.detect(img, confThreshold = 0.5)
    print(classIds, bboxes)


    #indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.5)

    if len(classIds) != 0:
        for classId, conf , bbox in zip(classIds.flatten(), confs.flatten(),bboxes):
            cv2.rectangle(img, bbox, color = (0,255,0),thickness = 2)
            cv2.putText(img, classNames[classId - 1], (bbox[0] + 10, bbox[1] +30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)
            if conf > 0.50 and conf < 0.70:
                cv2.putText(img, f"{round(conf*100,2)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,165),1)
            else:
                cv2.putText(img, f"{round(conf*100,2)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    cv2.imshow("object",img)
    cv2.waitKey(1)
    # print(net)
    # print(classNames)
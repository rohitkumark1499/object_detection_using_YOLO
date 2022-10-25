import cv2
import numpy as np

# we are not going to bother with objects less than 30% probability
THRESHOLD = 0.3
# the lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.3
YOLO_IMAGE_SIZE = 320


def find_objects(model_outputs):
    ############## NMS ###############
    # initialization for non-max suppression (NMS)
    # declare list for [class id], [box center, width & height], [confidences]
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    ############## NMS ###########

    for object_detection_layer in model_outputs:
        for object_detection in object_detection_layer:

            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            # take only predictions with confidence more than 20%
            if prediction_confidence > 0.20:
                predicted_class_label = classes[predicted_class_id]
                # obtain the bounding box co-oridnates for actual image from blob image
                bounding_box = object_detection[0:4] * np.array([original_width, original_height, original_width, original_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))

                ############## NMS ###############
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                ############## NMS ###########

    ############## NMS ###############

    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    for max_valueid in max_value_ids:
        max_class_id = max_valueid
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = classes[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]

        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        box_color = class_colors[predicted_class_id]

        box_color = [int(c) for c in box_color]

        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))

        cv2.rectangle(image, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(image, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

image = cv2.imread('Images/scene3.jpg')
original_width, original_height = image.shape[1], image.shape[0]

#coco dataset labels
classes = []
with open('requirements/coco-labels-2014_2017.txt', 'r') as f:
  classes=[line.strip() for line in f.readlines()]
print(classes)

class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors,(16,1))

neural_network = cv2.dnn.readNetFromDarknet('/home/rohit/PycharmProjects/computer_vision_project/Object_detection_with_YOLO/requirements/yolov3.cfg',
                                            '/home/rohit/PycharmProjects/computer_vision_project/Object_detection_with_YOLO/requirements/yolov3.weights')
# define whether we run the algorithm with CPU or with GPU
# WE ARE GOING TO USE CPU !!!
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# the image into a BLOB [0-1] RGB - BGR
blob = cv2.dnn.blobFromImage(image, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
neural_network.setInput(blob)

layer_names = neural_network.getLayerNames()
# YOLO network has 3 output layer - note: these indexes are starting with 1
output_names = [layer_names[index - 1] for index in neural_network.getUnconnectedOutLayers()]

outputs = neural_network.forward(output_names)
find_objects(outputs)


cv2.imshow('YOLO Algorithm', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
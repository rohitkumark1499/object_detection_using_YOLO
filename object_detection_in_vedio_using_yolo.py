import numpy as  np
import cv2

file_video_stream = cv2.VideoCapture('videos/objects.mp4')
class_labels = []
with open('requirements/coco-labels-2014_2017.txt', 'r') as f:
  class_labels=[line.strip() for line in f.readlines()]
print(class_labels)

while True:
    ret, current_frame = file_video_stream.read()
    img_to_detect = current_frame

    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]

    # convert to blob to pass into model
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)

    class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors, (16, 1))

    yolo_model = cv2.dnn.readNetFromDarknet('/home/rohit/PycharmProjects/computer_vision_project/Object_detection_with_YOLO/requirements/yolov3.cfg',
                                            '/home/rohit/PycharmProjects/computer_vision_project/Object_detection_with_YOLO/requirements/yolov3.weights')

    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    yolo_model.setInput(img_blob)
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    for object_detection_layer in obj_detection_layers:
        for object_detection in object_detection_layer:

            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            if prediction_confidence > 0.20:
                predicted_class_label = class_labels[predicted_class_id]
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    for max_valueid in max_value_ids:
        max_class_id = max_valueid
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]

        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        box_color = class_colors[predicted_class_id]

        box_color = [int(c) for c in box_color]

        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))

        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    box_color, 1)

    cv2.imshow('Yolo algorithm', img_to_detect)

    if cv2.waitKey(1) == ord('q'):
        break

file_video_stream.release()
cv2.destroyAllWindows()
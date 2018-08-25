from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import numpy as np
from keras.models import load_model

class TLDetection():

    def __init__(self):
        self.detection_graph = self.load_graph('light_classification/model_detect.pb')        
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.detection_number = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def filter_boxes(self, min_score, boxes, scores, classes):
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and classes[i] == 10: # traffic light class == 10
                idxs.append(i)
    
        filt_boxes = boxes[idxs, ...]
        filt_scores = scores[idxs, ...]
        filt_classes = classes[idxs, ...]
        return filt_boxes, filt_scores, filt_classes

    def run(self, image):
        # Run detection
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.detection_number], 
                                        feed_dict={self.image_tensor: image})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with only traffic lights
        return self.filter_boxes(0.05, boxes, scores, classes)

    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

class TLClassifier(object):
    def __init__(self, is_site):
        if is_site: # load either classifier for carla or for hte sim
            self.classifier = load_model('light_classification/classifier_carla.h5')
        else:    
            self.classifier = load_model('light_classification/classifier_sim.h5')
        self.graph = tf.get_default_graph()            
        self.detection = TLDetection()     
        
    def get_classification(self, image):

        img_expanded = np.expand_dims(np.asarray(image, dtype=np.uint8), 0) 
        
        # Get detection boxes, scores and classes for traffic lights
        boxes, scores, classes = self.get_traffic_lights(img_expanded)

        if len(scores) > 0: # check that a detection was made
            max_ind = scores.tolist().index(max(scores)) # get the best prediction
            height, width, channels = image.shape       
            box_coords = self.to_image_coords(boxes[max_ind], height, width)

            ymin = int(box_coords[0])
            xmin = int(box_coords[1])
            ymax = int(box_coords[2])
            xmax = int(box_coords[3])
            
            image_a = np.asarray(image)
            cropped_image = image_a[max(ymin-20,0):min(ymax+20,height), max(xmin-20,0):min(xmax+20,width), :]
            image_resized = cv2.resize(cropped_image, (32, 32))

            image_resized = image_resized/255.0-0.5

            light_color = self.get_light_classification(image_resized, 32, 32, 3)

            if light_color == 0:
                return TrafficLight.RED
            elif light_color == 1:
                return TrafficLight.YELLOW
            elif light_color == 2:
                return TrafficLight.GREEN
            else:
                return TrafficLight.UNKNOWN
            
        return TrafficLight.UNKNOWN
    

    def to_image_coords(self, boxes, height, width):
        """
            Convert coordinates based on the image size.
        """
        box_coords = np.zeros_like(boxes)    
        box_coords[0] = boxes[0] * height
        box_coords[1] = boxes[1] * width
        box_coords[2] = boxes[2] * height
        box_coords[3] = boxes[3] * width

        return box_coords 

    def get_traffic_lights(self, image):
        return self.detection.run(image)


    def get_light_classification(self, image, height, width, channels):
        with self.graph.as_default():
            predictions = self.classifier.predict(image.reshape((1, height, width, channels)))
            color =  predictions[0].tolist().index(np.max(predictions[0]))
            return color       

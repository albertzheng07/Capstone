from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
import rospkg

class TLClassifier(object):
    def __init__(self, is_site, width, height, channels=3):
        self.width = width
        self.height = height
        self.channels = channels

        r = rospkg.RosPack()
        path = r.get_path('tl_detector')
        
        if is_site:
            self.model = load_model(path + '/light_classification/classifier_carla.h5')
        else:    
            self.model = load_model(path + '/light_classification/classifier_sim.h5')    
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        resized = cv2.resize(image, (self.width, self.height))
        resized = resized / 255.;

        with self.graph.as_default():
            predictions = self.model.predict(resized.reshape((1, self.height, self.width, self.channels)))
            color =  predictions[0].tolist().index(np.max(predictions[0])) # get color prediction

            rospy.logwarn("predictions ",predictions)
            rospy.logwarn("predictions[0]",predictions[0])            
            rospy.logwarn("color = ",color)

            if color == TrafficLight.RED: 
                return TrafficLight.RED        

        return TrafficLight.UNKNOWN

#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float64
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math
import numpy as np
from styx_msgs.msg import Lane
from twist_controller import Controller
from geometry_msgs.msg import PoseStamped

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)
        self.cte_pub = rospy.Publisher('/cte',
                                         Float64, queue_size=1)
        
        # TODO: Create `Controller` object
        self.controller = Controller(vehicle_mass = vehicle_mass,
                                     fuel_capacity = fuel_capacity,
                                     brake_deadband = brake_deadband,
                                     decel_limit = decel_limit,
                                     accel_limit = accel_limit,
                                     wheel_radius = wheel_radius,
                                     wheel_base = wheel_base,
                                     steer_ratio  = steer_ratio,
                                     max_lat_accel = max_lat_accel,
                                     max_steer_angle = max_steer_angle)

        self.current_vel = None
        self.dbw_enabled = None
        self.linear_vel_cmd = None
        self.angular_vel_cmd = None
        self.cte = None
        self.throttle = self.steering = self.brake = 0
        self.current_pose = None
        self.final_waypoints = None

        # TODO: Subscribe to all the topics you need tos
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('final_waypoints', Lane, self.final_waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb, queue_size=1)
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            # if <dbw is enabled>:
            #   self.publish(throttle, brake, steer)
            if not None in (self.final_waypoints, self.current_pose):
                self.cte = self.get_cte(self.final_waypoints, self.current_pose)

            if not None in (self.current_vel, self.angular_vel_cmd, self.linear_vel_cmd, self.cte):
                self.steering, self.brake, self.throttle = self.controller.control(self.current_vel,
                                                                    self.dbw_enabled, 
                                                                    self.linear_vel_cmd,
                                                                    self.angular_vel_cmd,
                                                                    self.cte)
#                 rospy.loginfo("velocity error = %f", self.linear_vel_cmd-self.current_vel)
#                 rospy.loginfo("cte = %f", self.cte) 
                
            if self.dbw_enabled:
                #pass
                self.publish(self.throttle, self.brake, self.steering)

            rate.sleep()

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data

    def twist_cb(self, msg):
        self.linear_vel_cmd = msg.twist.linear.x
        self.angular_vel_cmd = msg.twist.angular.z

    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    def final_waypoints_cb(self, msg):
        self.final_waypoints = msg.waypoints

    def current_pose_cb(self, msg):
        self.current_pose = msg
 
    def get_cte(self, final_waypoints, current_pose):

        if len(final_waypoints) > 3:
            starting_wp = final_waypoints[0].pose.pose.position

            waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in final_waypoints]

            # Shift the points to the starting wp
            waypoints_2d_shift = waypoints_2d - np.array([starting_wp.x, starting_wp.y])

            # Derive the yaw from a set of the waypoints
            if len(final_waypoints) > 10:
                num_wpts = 10
            else:
                num_wpts = len(final_waypoints)/2
            wp_yaw = np.arctan2(waypoints_2d_shift[num_wpts, 1], waypoints_2d_shift[num_wpts, 0])
            rot_mat = np.array([
                    [np.cos(wp_yaw), -np.sin(wp_yaw)],
                    [np.sin(wp_yaw), np.cos(wp_yaw)]
                ])

            waypoints_heading = np.dot(waypoints_2d_shift, rot_mat)

            # Fit a polynomial to the set of the heading of the offset waypoint frame
            coefficients = np.polyfit(waypoints_heading[:, 0], waypoints_heading[:, 1], 2)

            # Transform the current pose of the car to be in the waypoint set's coordinate system
            shifted_pose = np.array([current_pose.pose.position.x - starting_wp.x, current_pose.pose.position.y - starting_wp.y])
            rotated_pose = np.dot(shifted_pose, rot_mat)

            path_y = np.polyval(coefficients, rotated_pose[0])
            cte =  path_y - rotated_pose[1]
        else:
            cte = 0

        return cte


    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)
        
        self.cte_pub.publish(self.cte)


if __name__ == '__main__':
    DBWNode()

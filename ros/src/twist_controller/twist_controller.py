
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import rospy

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
    			 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # initialize variables
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        min_speed = 0.1

        # Yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # throttle gains
        kp_thr = 0.2
        ki_thr = 0.01
        kd_thr = 0.1
        minThr = self.decel_limit
        maxThr = self.accel_limit*0.4
        # Throttle controller
        self.throttle_controller = PID(kp_thr, ki_thr, kd_thr, minThr, maxThr)

        kp_str = 0.2
        ki_str = 0.001
        kd_str = 0.2
        minStr = -max_steer_angle
        maxStr = max_steer_angle

        # Steering controller
        self.steering_controller = PID(kp_str, ki_str, kd_str, minStr, maxStr)

        # LP filter
        tau = 0.4 # 1/(2*pi*tau) = fcutoff
        ts = 0.02 # sample period
        self.vel_LPF = LowPassFilter(tau, ts)

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel, cte):
    	if not dbw_enabled:
            self.throttle_controller.reset()
            self.steering_controller.reset()
            return 0., 0., 0.
		
        current_vel = self.vel_LPF.filt(current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        dt =  current_time - self.last_time
        self.last_time = current_time

        steering = 0
        #steer_FF = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        steering = self.steering_controller.step(cte, dt) #+ steer_FF
        
        throttle = self.throttle_controller.step(vel_error, dt)

        brake = 0

		# # add dead band inputs
        if linear_vel == 0 and current_vel < 0.1:
            throttle = 0
            brake = 400 # torque input of 400 Nm to stop vehicle
        elif throttle < 0:
            decel = max(throttle, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # R * m * decel
            throttle = 0

        return steering, brake, throttle



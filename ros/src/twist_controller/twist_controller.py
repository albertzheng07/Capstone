
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

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
        kp = 0.2
        ki = 0.1
        kd = 0
        minThr = 0
        maxThr = 0.2
        # Thorttle controllre
        self.throttle_controller = PID(kp, ki, kd, minThr, maxThr)

        # LP filter
        tau = 0.5 # 1/(2*pi*tau) = fcutoff
        ts = 0.02 # sample period
        self.vel_LPF = LowPassFilter(tau, ts)

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
    	if not dbw_enabled:
    		self.throttle_controller.reset()
       		return 0., 0., 0.
		
		current_vel = self.vel_LPF.filt(current_vel)

		steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

		vel_error = linear_vel - current_vel
		self.last_vel = current_vel

		current_time = rospy.get_time()
		dt = self.last_time - current_time

		throttle = self.throttle_controller.step(vel_error, dt)

		brake = 0

		# add dead band inputs
		if linear_vel == 0 and current_vel < 0.1:
			throttle = 0
			brake = 400 # torque input of 400 Nm to stop vehicle
		elif throttle < 0.1 and vel_error < 0:	
			throttle = 0
			decel = max(vel_error, self.decel_limit)
			brake = abs(decel)*self.vehicle_mass*self.wheel_radius # R * m * decel

		return steering, brake, throttle




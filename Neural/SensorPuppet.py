from re import A
from webbrowser import get
from numpy.core.shape_base import block
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
from pynput.keyboard import Key, Listener
import rospy
import serial
import time
import math
import csv
import os
import Xenbartender
import keras
from joblib import dump, load
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

##<Nicolas>
CSV_HEADER = ['delta_time', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'magn_x', 'magn_y', 'magn_z', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4' ]
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
time_stmp = 0.0
delta_time = 0
model = keras.models.load_model('Puppet')
regression = load('regression_puppet.sav')
def read_data(timestamp):
	data = arduino.readline().decode('utf-8')[:-1]
	data = data.split(';')[0:10]
	return data
##</Nicolas>

def format_data(rawdata):
	while(len(rawdata) != 10):
		rawdata = read_data(0.05)
		print('<error>\n	sensor data is too small:')
		print(rawdata)
		print('</error>')
	
	format_error = True
	while format_error:
		try:
			data = [float(x) for x in rawdata]
			format_error = False
		except:
			rawdata = read_data(0.05)
			print('incorrect data format')
			print('new data:')
			print(data)
			format_error = True

	global delta_time
	global time_stmp
	delta_time = (data[0] - time_stmp) * .001
	time_stmp = data[0] 
	accel_x = data[1] * .1 * np.pi * .5
	accel_y = data[2] * .1 * np.pi * .5
	accel_z = data[3] * .1 * np.pi * .5
	gyro_x = data[4] * .001 * -1
	gyro_y = data[5] * .001
	gyro_z = data[6] * .001
	magn_x = data[7]
	magn_y = data[8]
	magn_z = data[9]

	return delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

last_command = [0, 0, 0, 0, 0]
def normalize(new_vect, threshold):
	global last_command
	for i, x in enumerate(new_vect):
		if(math.dist(new_vect, last_command) < threshold):
			new_vect[i] = last_command[i]
		else:
			last_command[i] = new_vect[i]
	return new_vect

def get_curr_vect(current):
	data = read_data(0.05)
	global delta_time
	delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z = format_data(data)
	joint_vector = [current[0] + gyro_x, current[1] + gyro_y, current[2] + gyro_z * .8,  current[2] + gyro_z * .9, 0.0]
	joint_vector = normalize(joint_vector, .01)
	full_vector = [delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z]
	print("raw data:")
	print(data)
	print("Caught joint_vector:")
	print(joint_vector)
	return joint_vector, full_vector

def is_data_valide(data):
	if(len(data) != 5):
		print("data is missing joints")
		return False
	if not all(isinstance(item, float) for item in data):
		print("data is not in the correct format (float)")
		return False
	return True

def dct(row):
	dictionary = {'delta_time' : 0.0}
	for indx,key in enumerate(CSV_HEADER):
		dictionary[key] = row[indx]
	return dictionary


def arm_loop(save_csv, writer):
	bot = InterbotixManipulatorXS("px150", "arm", "gripper")
	bot.arm.go_to_home_pose()
	while not rospy.is_shutdown():
			try:
				current = bot.arm.get_joint_commands()
				joints,full = get_curr_vect(current)
				joints  = [float(x) for x in joints]
				save_vector = np.zeros(10)
				if(is_data_valide(joints)):
					bot.arm.publish_positions(joints, blocking = False, moving_time= delta_time, accel_time=0)
					save_vector = full + current
					if(save_csv):
						writer.writerow(dct(save_vector))
				else:
					print("invalid data")
			except KeyboardInterrupt:
				break
	print("Executing shutdown routine...")
	Xenbartender.use()

def arm_loop_infer(inference_func, save_csv, writer):
	bot = InterbotixManipulatorXS("px150", "arm", "gripper")
	bot.arm.go_to_home_pose()
	while not rospy.is_shutdown():
			try:
				current = bot.arm.get_joint_commands()
				joints,full = inference_func(current)
				joints  = [float(x) for x in joints]
				save_vector = np.zeros(10)
				if(is_data_valide(joints)):
					print("data is valid ---- PUBLISHING..../")
					print(joints)
					bot.arm.publish_positions(joints, blocking = False, moving_time= delta_time, accel_time= delta_time)
					save_vector = full + joints
					if(save_csv):
						writer.writerow(dct(save_vector))
				else:
					print("invalid data")
			except KeyboardInterrupt:
				break
	print("Executing shutdown routine...")
	Xenbartender.use()

def predict_joints(current):
	data = read_data(0.05)
	delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z = format_data(data)
	X_test = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
	# magn_x, magn_y, magn_z
	joints = regression.predict(np.asarray([X_test]))[0]
	joints = np.asarray(joints)
	print(f"{bcolors.WARNING}prdcted:")
	print(joints)
	print(f"{bcolors.WARNING}/prdcted:")
	#joints = normalize(joints, .01)
	return joints, X_test

def exec_puppet(save_csv):
	if save_csv:
		with open('SensorPuppet.csv', 'w', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
			writer.writeheader()
			arm_loop_infer(inference_func=get_curr_vect, save_csv=True, writer=writer)
	else:
		arm_loop(save_csv=False, writer=None)

	print("ShuttingDown...")
	
def exec_puppet_infer():
	arm_loop_infer(inference_func=predict_joints, save_csv=False, writer=None)
	

def main():
	exec_puppet(False)
	#exec_puppet_infer()
	""""
	opt = input("do you want to use neural inference ?\n1 = yes.\n2 = no.\n")
	if opt == "1":
		exec_puppet_infer()
	elif opt == "2":
		opt = input("do you want to save data ?\n1 = yes.\n2 = no.\n")
		if opt == "1":
			exec_puppet(True)
		elif opt == "2":
			exec_puppet(False)
		else:
			print("ERROR: invalid answer")
			main()
	else:
		print("ERROR: invalid answer")
		main()
	"""

if __name__=='__main__':
	main()

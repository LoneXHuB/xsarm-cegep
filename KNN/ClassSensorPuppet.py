from email import header
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
import keras
from joblib import dump, load
import pickle

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

CLASS_LENGTH = 10
CSV_HEADER = ['accel_x', 'accel_y', 'accel_z']
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
time_stmp = 0.0
delta_time = 0

filename = 'finalized_model.sav'
skit_classifier = pickle.load(open(filename, 'rb'))

def read_data(timestamp):
    time.sleep(timestamp)
    data = arduino.readline().decode('utf-8')[:-1]
    data = data.split(';')[0:10]
    return data

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

def dct(row, csv_header):
	dictionary = {}
	for indx,key in enumerate(csv_header):
		dictionary[key] = row[indx]
	return dictionary


def arm_loop(save_csv, writer):

    print("CSV HEADER :")
    print(CSV_HEADER)
    while True:
        try:
            opt= input("give batch value [0:4] or enter q to quit")
            if opt == 'q': break
            opt = int(opt)
            if opt < 0: raise Exception("opt out of range")
            if opt > 4: raise Exception("opt out of range")

            print("writing with opt:" + str(opt))
            data = read_data(0.01)
            delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z = format_data(data)
            save_vector = [accel_x, accel_y, accel_z]
            for i in range(CLASS_LENGTH):
                data = read_data(0.01)
                delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z = format_data(data)
                save_vector += [accel_x, accel_y, accel_z]
            save_vector += [opt]
            if(save_csv):   
                print("wrote line")
                print(save_vector)
                writer.writerow(dct(save_vector, CSV_HEADER))
        except:
            print("invalid mov value: "+ str(opt))

    
def predict_joints():
    data = read_data(0.01)
    delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z = format_data(data)
    batch = [accel_x, accel_y, accel_z]
    for i in range(CLASS_LENGTH):
        data = read_data(0.01)
        delta_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magn_x, magn_y, magn_z = format_data(data)
        batch += [accel_x, accel_y, accel_z]
	# magn_x, magn_y, magn_z
    pred = skit_classifier.predict(np.asarray([batch]))[0]
    return pred

def arm_loop_infer():
	bot = InterbotixManipulatorXS("px150", "arm", "gripper")
	while not rospy.is_shutdown():
            try:
                pred = predict_joints()
                print("detected motion : " + pred)
                pred = int(pred)
                print("data is valid ---- PUBLISHING..../")
                if pred == 0:
                    print("movement zero detected... going to home pose...////")
                    bot.arm.go_to_home_pose()
                elif pred == 1:
                    joints=[np.pi, 0.0 , 0.0 , 0.0 ,0.0]
                elif pred == 2:
                    joints=[np.pi/2, -np.pi /2, 0.0 , 0.0 ,0.0]
                elif pred == 3:
                    joints=[-np.pi/2, -np.pi /2, np.pi / 4 , 0.0 ,0.0]
                elif pred == 4:
                    joints=[np.pi/2, 0.0 , -np.pi /2, 0.0 , 1]
                else:
                    bot.arm.go_to_sleep_pose()
                if pred != 0:
                    bot.arm.publish_positions(joints, blocking = False, moving_time= 1, accel_time= 0.2)
                
            except :
                print("ERROR : Executing shutdown routine...")
                bot.arm.go_to_home_pose()

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
	
def exec_write_class_data():
    global CSV_HEADER
    h = np.asarray(CSV_HEADER).astype('U10')
    final_header = np.resize(h, CLASS_LENGTH * len(CSV_HEADER) + len(CSV_HEADER))
    final_header = np.append(final_header, 'move')
    
    batch_count = 0
    for indx,x in enumerate(final_header):
        if indx % len(CSV_HEADER) == 0 :
            batch_count += 1
        final_header[indx] = x + str(batch_count)
    
    CSV_HEADER = final_header
    
    with open('ClassSensorPuppet.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_header)
        writer.writeheader()
        arm_loop(save_csv=True, writer=writer)

def main():
    opt = input("Menu :\n 1-Gesture recognition control\n2-fill train data\n")
    if opt == '1':
        arm_loop_infer()
    elif opt == '2':
        exec_write_class_data()
    else: main()


if __name__=='__main__':
	main()

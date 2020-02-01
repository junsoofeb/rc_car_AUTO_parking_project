#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from Tkinter import *
from socket import *     
from imutils.video import FPS
import imutils
import time
import cv2 as cv
import numpy as np
import random

ctrl_cmd = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']

# my raspberry pi IP
HOST = '172.30.1.14' 
#HOST = '172.30.1.25' 
PORT = 21567
BUFSIZ = 1024             # buffer size
ADDR = (HOST, PORT)

#tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
#tcpCliSock.connect(ADDR)                    # Connect with the server

def forward(event):
	print('forward')
	tcpCliSock.send('forward')

def backward(event):
	print('backward')
	tcpCliSock.send('backward')

def handle_left(event):
	print('handle left')
	tcpCliSock.send('left')

def handle_right(event):
	print('handle right')
	tcpCliSock.send('right')

def stop(event):
	print('stop')
	tcpCliSock.send('stop')

def home(event):
	print('home')
	tcpCliSock.send('home')

# 소켓 통신 종료
def quit(event):
	tcpCliSock.send('stop')
	tcpCliSock.close()

'''
top.bind('<KeyPress-a>', left_fun)   # Press down key 'A' on the keyboard and the car will turn left.
top.bind('<KeyPress-d>', right_fun) 
top.bind('<KeyPress-s>', backward_fun)
top.bind('<KeyPress-w>', forward_fun)
top.bind('<KeyPress-h>', home_fun)
top.bind('<KeyRelease-a>', home_fun) # Release key 'A' and the car will turn back.
top.bind('<KeyRelease-d>', home_fun)
top.bind('<KeyRelease-s>', stop_fun)
top.bind('<KeyRelease-w>', stop_fun)
'''

# 상태변환 확률
TRANSITION_PROB = 1
# 6가지 행동 : 전진, 대각선 전진 2, 후진, 대각선 후진 2
POSSIBLE_ACTIONS = [0, 1, 2, 3, 4, 5, 6]  
REWARDS = []
GOAL = None
# 전체 환경 크기
H = 480
W = 640
#REWARDS = [[0 for col in range(W)] for row in range(H)]
# 전부다 0으로 초기화
REWARDS = np.zeros((H,W))


# 행동에 따른 보상 반환
def get_reward(state, action):
    next_state = state_after_action(state, action)

    return reward[next_state[0]][next_state[1]]

def state_after_action(state, action_index):
    action = ACTIONS[action_index]

    return check_boundary([state[0] + action[0], state[1] + action[1]])

def check_boundary(state):
    state[0] = (0 if state[0] < 0 else W - 1
                if state[0] > W - 1 else state[0])
    state[1] = (0 if state[1] < 0 else H - 1
                if state[1] > H - 1 else state[1])
    return state

def get_transition_prob( state, action):
    return transition_probability

def get_all_states():
    return all_state



frame = None

# 아래 2줄의 G_img와 G_copy는 단순히 초기화 용도! 
G_img = np.zeros((1,1))
G_copy = G_img.copy()
moues_pressed = False
s_x = s_y  = e_x = e_y = -1
def mouse_callback(event, x, y, flags, param):
        global G_img, G_copy, s_x, s_y, e_x, e_y, moues_pressed
        if event == cv.EVENT_LBUTTONDOWN:
            moues_pressed = True
            s_x, s_y = x, y
            G_img = G_img.copy()

        elif event == cv.EVENT_MOUSEMOVE:
            if moues_pressed:
                G_copy = G_img.copy()
                cv.rectangle(G_copy, (s_x, s_y), (x, y), (0, 255, 0), 3)

        elif event == cv.EVENT_LBUTTONUP:
            moues_pressed = False
            e_x, e_y = x, y


parking_zone = None
# 주차장 위치를 마우스로 드래그 한 뒤 'w'
def set_parkinkg_zone(img):
    global G_img, G_copy, s_x, s_y, e_x, e_y, moues_pressed, frame, parking_zone, GOAL
    w, h = img.shape[0], img.shape[1]
    G_copy = img.copy()
    G_img = img.copy()
    cv.namedWindow("Press 'w' to Set PARKING ZONE!")
    cv.setMouseCallback("Press 'w' to Set PARKING ZONE!", mouse_callback)    
    while True:
        cv.imshow("Press 'w' to Set PARKING ZONE!", G_copy)
        key = cv.waitKey(1)
        if key == ord('w'):
            if s_y > e_y:
                s_y, e_y = e_y, s_y
            if s_x > e_x:
                s_x , e_x = e_x, s_x
            if e_y - s_y > 1 and e_x - s_x > 0:
                parking_zone = [(s_x, s_y), (e_x, e_y)] # [(좌상단), (우하단)]
                x = parking_zone[0][0] + (parking_zone[1][0] - parking_zone[0][0])//2
                y = parking_zone[0][1] + (parking_zone[1][1] - parking_zone[0][1])//2
                GOAL = (x, y)
                break
        # 박스 그리기 실수 했을 경우 esc누르면 다시 시작.
        elif key == 27:
            G_copy = G_img.copy()
            continue
    cv.destroyAllWindows()
    





# 주차선 넘어가면 패널티 받도록, 주차선 위치에 패널티 부여! 
# 단, 주자할때 들어가는 한 면은 제외!!
cam = 0
def set_penalty():
    global cam, REWARDS
    cap = cv.VideoCapture(cam)
    while True:
        _, frame = cap.read()
        set_parkinkg_zone(frame)
        break
    cap.release()
    cv.destroyAllWindows()
    
    '''
    # parking_zone == [(좌상단), (우하단)]
    min_x = parking_zone[0][0]
    max_x = parking_zone[1][0]
    min_y = parking_zone[0][1]
    max_y = parking_zone[1][1]

    print(min_x, min_y, max_x, max_y)
    for x in range(min_x, max_x + 1):
        REWARDS[x][min_y] = -10 # 페널티 부여, 맨 위 

    for y in range(min_y, max_y + 1):
        REWARDS[min_x][y] = -10 # 페널티 부여, 왼쪽 벽
        REWARDS[max_x][y] = -10 # 페널티 부여, 오른쪽 벽
    '''   
    

first_location = None
first_box = None
initial_box = None
def set_start_state():
    global cam, first_location, first_box, initial_box
    tracker = cv.TrackerMedianFlow_create()
    fps = None
    cap = cv.VideoCapture(cam)
    while True:
        _, frame = cap.read()
        # 첫 박스가 지정되었으면,
        if initial_box is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                first_box = (x, y, w, h)
                first_location = (x + w//2, y + h//2)
                cv.circle(frame, first_location, 3, (0,0,255),3)
            fps.update()
            fps.stop()
            cv.imshow("current location, 'q' to FINISH!", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break 
        else:
            # 첫 박스 지정하는 과정
            cv.imshow("Press 'w' to select ROI, 'enter' to FINISH!", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("w"):
                initial_box = cv.selectROI("Press 'w' to select ROI, 'enter' to FINISH!", frame, fromCenter=False, showCrosshair=True)
                tracker.init(frame, initial_box)
                fps = FPS().start()
                cv.destroyAllWindows()
            elif key == ord("q"):
                break 
    cap.release() 
    cv.destroyAllWindows()
    
def get_state():
    '''
    first_box = (x, y, w, h)
    first_location = (x + w//2, y + h//2)
    parking_zone = [(s_x, s_y), (e_x, e_y)] # [(좌상단), (우하단)]
    '''
    global cam, first_box, first_location, parking_zone, initial_box, REWARDS, GOAL
    set_start_state()
    set_penalty()
    # parking_zone == [(좌상단), (우하단)]
    min_y = parking_zone[0][0]
    max_y = parking_zone[1][0]
    min_x = parking_zone[0][1]
    max_x = parking_zone[1][1]

    for x in range(min_x, max_x + 1):
        REWARDS[x][min_y] = 255 
        REWARDS[x][max_y] = 255
    for y in range(min_y, max_y + 1):
        REWARDS[min_x][y] = 255 


    '''
    OPENCV_OBJECT_TRACKERS 종류
    "csrt": cv.TrackerCSRT_create()
    "kcf": cv.TrackerKCF_create()
    "boosting": cv.TrackerBoosting_create()
    "mil": cv.TrackerMIL_create()
    "tld": cv.TrackerTLD_create()
    "medianflow": cv.TrackerMedianFlow_create()
    "mosse": cv.TrackerMOSSE_create()
    종류별 특징보고 선택
    '''
    tracker = cv.TrackerMedianFlow_create()
    fps = None
    cap = cv.VideoCapture(cam)
    temp = ()
    _,frame = cap.read()
    
    tracker.init(frame, initial_box)
    while True:
        R_copy = REWARDS.copy()
        _, frame = cap.read()
        fps = FPS().start()
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            temp = (x,y, x+w, y+h)
            first_box = (x, y, w, h)
            first_location = (x + w//2, y + h//2)
            cv.circle(frame, first_location, 3, (0,0,255),3)
        fps.update()
        fps.stop()

        cv.rectangle(frame, parking_zone[0], parking_zone[1], (255, 0, 0), 2)

        cv.circle(frame, GOAL ,5, (0,0,255), -1)
        
        cv.circle(R_copy, first_location, 5, 255, -1)
        cv.circle(R_copy, GOAL, 5, 255, -1)
        cv.rectangle(R_copy, (temp[0], temp[1]), (temp[2], temp[3]) , 255, 2)

        
        cv.imshow("env", frame)
        cv.imshow("movement tracking", R_copy)
        print("current state:", first_location)
        key = cv.waitKey(30) & 0xFF
        if key == ord("s"):
            break
    cap.release()
    cv.destroyAllWindows()


get_state()

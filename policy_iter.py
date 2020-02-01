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

tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)                    # Connect with the server

def forward():
	print('forward')
	tcpCliSock.send(b'forward')

def backward():
	print('backward')
	tcpCliSock.send(b'backward')

def handle_left():
	print('handle left')
	tcpCliSock.send(b'left')

def handle_right():
	print('handle right')
	tcpCliSock.send(b'right')

def stop():
	print('stop')
	tcpCliSock.send(b'stop')

def home():
	print('home')
	tcpCliSock.send(b'home')

# 소켓 통신 종료
def quit():
	tcpCliSock.send(b'stop')
	tcpCliSock.close()


# **********************************************************************************
#                                   ENVIRONMENT
# **********************************************************************************


# state 는 (x,y) 좌표 정보! state[0], state[1]

# 상태변환 확률
TRANSITION_PROB = 1
# 6가지 행동 : 전진, 대각선 전진 2, 후진, 대각선 후진 2
POSSIBLE_ACTIONS = [0, 1, 2, 3, 4, 5]  
REWARDS = []
GOAL = None
# 전체 환경 크기
H = 480
W = 640
#REWARDS = [[0 for col in range(W)] for row in range(H)]
# 전부다 0으로 초기화
REWARDS = np.zeros((H,W))
ALL_STATE = REWARDS.copy()
CURRNET_STATE = None
return_flag = False
# 행동에 따른 보상 반환
def get_reward(state, action):
    next_state = state_after_action(state, action)
    # next_state는 (x, y)
    print(next_state)

    return REWARDS[int(next_state[0]])[int(next_state[1])]

def return_state(location):
    global CURRENT_STATE, return_flag

    CURRENT_STATE = location
    return_flag = False
    
    return CURRENT_STATE

# 행동을 취한 뒤의 state
def state_after_action(state, action):    
    global POSSIBLE_ACTIONS, return_flag
    time_gap = 0.5
    if action == 0:
        forward()
        time.sleep(time_gap)
        stop()
    elif action == 1:
        handle_left()
        forward()
        time.sleep(time_gap)
        home()
        stop()
    elif action == 2:
        handle_right()
        forward()
        time.sleep(time_gap)
        home()
        stop()
    elif action == 3:
        backward()
        time.sleep(time_gap)
        stop()
    elif action == 4:
        handle_left()
        backward()
        time.sleep(time_gap)
        home()
        stop()
    else:
        handle_right()
        backward()
        time.sleep(time_gap)
        home()
        stop()
    
    time.sleep(1)
    return_flag = True
    next_state = return_state(state)
        
    return next_state


def get_transition_prob(state, action):
    global TRANSITION_PROB

    return TRANSITION_PROB

def get_all_states():
    global ALL_STATE

    return ALL_STATE





# **********************************************************************************
#                                   Policy_Iter
# **********************************************************************************
# 가치함수를 2차원 리스트로 초기화
value_table = [[0.0] * W for _ in range(H)]
value_table = np.array(value_table)
# 모든 행동에 대해 동일한 확률로 정책 초기화
policy_table = [[[0.25, 0.25, 0.25, 0.25, 0.25, 0.25]] * W for _ in range(H)]
policy_table = np.array(policy_table)
# 마침 상태의 설정
x = None
y = None
# 감가율
discount_factor = 0.9
def policy_evaluation():
    global x, y, POSSIBLE_ACTIONS
    # 다음 가치함수 초기화
    next_value_table = [[0.00] * W for _ in range(H)]
    next_value_table = np.array(next_value_table)
    # 모든 상태에 대해서 벨만 기대방정식을 계산
    for state in get_all_states():
        value = 0.0
        # 마침 상태의 가치 함수 = 0
        if state == [y, x] or state == (y, x):
            next_value_table[state[0]][state[1]] = value
            continue
        # 벨만 기대 방정식
        for action in POSSIBLE_ACTIONS:
            next_state = state_after_action(state, action)
            reward = get_reward(state, action)
            next_value = get_value(next_state)
            value += (get_policy(state)[action] *
                      (reward + discount_factor * next_value))
        next_value_table[state[0]][state[1]] = round(value, 2)
    value_table = next_value_table

# 현재 가치 함수에 대해서 탐욕 정책 발전
def policy_improvement():
    global x, y
    next_policy = policy_table
    for state in get_all_states():
        if state == [y, x] or state == (y, x):
            continue
        value = -99999
        max_index = []
        # 반환할 정책 초기화
        result = [0.0, 0.0, 0.0, 0.0]
        # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
        for index, action in enumerate(possible_actions):
            next_state = state_after_action(state, action)
            reward = get_reward(state, action)
            next_value = get_value(next_state)
            temp = reward + discount_factor * next_value
            # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
            if temp == value:
                max_index.append(index)
            elif temp > value:
                value = temp
                max_index.clear()
                max_index.append(index)
        # 행동의 확률 계산
        prob = 1 / len(max_index)
        for index in max_index:
            result[index] = prob
        next_policy[state[0]][state[1]] = result
    policy_table = next_policy

# 특정 상태에서 정책에 따른 행동을 반환
def get_action(state):
    # 0 ~ 1 사이의 값을 무작위로 추출
    random_pick = random.randrange(100) / 100
    policy = get_policy(state)
    policy_sum = 0.0
    # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
    for index, value in enumerate(policy):
        policy_sum += value
        if random_pick < policy_sum:
            print('action index:', index)
            return index
        
# 상태에 따른 정책 반환
def get_policy(state):
    if state == [y, x] or state == (y, x):
        return 0.0
    return policy_table[state[0]][state[1]]
# 가치 함수의 값을 반환
def get_value( state):
    # 소숫점 둘째 자리까지만 계산
    return round(value_table[state[0]][state[1]], 2)



# **********************************************************************************
#                                   ENV_function
# **********************************************************************************
def move_by_policy():
    pass





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
    global G_img, G_copy, s_x, s_y, e_x, e_y, moues_pressed, frame, parking_zone, GOAL, REWARDS, policy_table, x, y
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
                temp_x = parking_zone[0][0] + (parking_zone[1][0] - parking_zone[0][0])//2
                temp_y = parking_zone[0][1] + (parking_zone[1][1] - parking_zone[0][1])//2
                GOAL = (temp_x, temp_y)
                x, y = GOAL
                policy_table[y][x] = np.array([None, None, None, None, None, None])
                REWARDS[y][x] = 10
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
    global cam, first_box, first_location, parking_zone, initial_box, REWARDS, GOAL, return_flag
    set_start_state()
    set_penalty()
    # parking_zone == [(좌상단), (우하단)]
    min_y = parking_zone[0][0]
    max_y = parking_zone[1][0]
    min_x = parking_zone[0][1]
    max_x = parking_zone[1][1]
    R_copy = REWARDS.copy()

    for x in range(min_x, max_x + 1):
        REWARDS[x][min_y] = -10 
        REWARDS[x][max_y] = -10 
        R_copy[x][min_y] = 255
        R_copy[x][max_y] = 255
        
    for y in range(min_y, max_y + 1):
        REWARDS[min_x][y] = -10 
        R_copy[min_x][y] = 255 


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
        RR_COPY = R_copy.copy()
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
        
        cv.circle(RR_COPY, first_location, 5, 255, -1)
        cv.circle(RR_COPY, GOAL, 5, 255, -1)
        cv.rectangle(RR_COPY, (temp[0], temp[1]), (temp[2], temp[3]) , 255, 2)

        
        cv.imshow("env", frame)
        cv.imshow("movement tracking", RR_COPY)
        print("current state:", first_location)
        
        if return_flag:
            print("return location:", first_location)
            return_state(first_location)
        
        key = cv.waitKey(30) & 0xFF
        if key == ord("s"):
            break
    cap.release()
    cv.destroyAllWindows()

# agent의 초기 상태 및 목표 위치 설정
get_state()

# 정책 반복 시작

# 정책 평가
policy_evaluation()
# 정책 발전
policy_improvement()
    
    

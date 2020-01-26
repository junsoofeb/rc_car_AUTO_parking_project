## default_files 


## 1. raspberry pi 초기 설정하는 법

### 0. ssh로 자신의 pi에 원격접속한다.  

ex) 터미널에서,    
ssh pi@IP ADDRESS    
보통 처음 pwd는 raspberry이다.  

### 1. 코드를 샘플코드를 복사해온다.

git clone https://github.com/sunfounder/Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi

### 2. 자신의 rc car의 정확한 조종을 위해서 calibration을 실행한다.

#### 터미널을 2개 실행하고, server 먼저 실행한다.  
#### client.py파일에서 IP_ADDRESS는 자신의 IP_ADDRESS로 수정!

#### Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/server/에서 
python cali_server.py

#### Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/client/에서  
python cali_client.py

calibration을 수행하고, 마치면 confirm을 눌러 종료.

#### 만약, servo가 발열이 심하고 동작을 하지 않는다면 재 조립이 필요! 
#### 조립할 때 servo의 최대 또는 최소 각도를 벗어난 채로 조립했기 때문!

### 3. RC CAR_Keyboard control 해보기


#### Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/server/에서 
python tcp_server.py

#### Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/client/에서  
python client_app.py

#### w,a,s,d 키를 이용해서 조종가능

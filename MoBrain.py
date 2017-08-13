import cv2
import numpy as np
import serial
import time
'''
自身功能：
    采集数据
        图像识别激光点
        读取下位机角度
被调用功能：
    目标检测
        颜色空间
        二值化
        形状匹配
    手写字符识别
        调用外部返回的坐标
'''
global ser
def openSerial():
    global ser
    ser = serial.Serial()
    ser.port = 'COM3'
    ser.baudrate = 112500
    ser.open()
    time.sleep(2)  # 等端口打开，官网说的!!
    pass
def readSTM():
    global ser
    ser.write(bytes(b'1'))
    data = ser.readline().decode("utf-8")
    print(type(data))
    # else:
    #     print 'no can'
    sig = data.split('\n')[0].split(' ')
    # sig[0]=float(sig[0])
    # sig[1]=float(sig[1])
    return sig

def ok():
    # 蓝色或红色hsv值
    mask_low=np.array([0,100,60])
    mask_high=np.array([20,255,180])
    # mask_low=np.array([])
    # mask_high=np.array([])
    cap=cv2.VideoCapture(0)
    fp=open('data.txt','a')
    while True:
        x = [0, 0]
        ret,frame=cap.read()
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # 根据阈值构建掩模
        res = cv2.inRange(hsv, mask_low, mask_high)
        # 对原图像和掩模进行位运算
        # res = cv2.bitwise_and(frame, frame, mask=mask)
        # res=cv2.cvtColor(cv2.cvtColor(mask,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
        # print(mask.shape)
        # res=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        # res = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
        # 阈值一定要设为 0！
        ret, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = cv2.Canny(res, 50, 150, apertureSize=3)
        #############################
        # minLineLength = 2
        # maxLineGap = 3
        # lines = cv2.HoughLinesP(res, 1, np.pi / 180, 7, minLineLength, maxLineGap)
        # try:
        #     for x1, y1, x2, y2 in lines[0]:
        #         cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # except:
        #     pass
        ############################
        circles = cv2.HoughCircles(res, cv2.HOUGH_GRADIENT, dp=1, minDist=3,
                                   param1=4, param2=4, minRadius=0, maxRadius=5)
        try:
            circles = np.uint16(np.around(circles))
            circles=circles[0, :]
            # print(len(circles))
            if(len(circles)==1):
                # print('hei')
                for i in circles:
                    # print('aa')
                    # draw the outer circle
                    # cv2.circle(frame, (i[0], i[1]), i[2], (255,0 , 0), 2)
                    # draw the center of the circle
                    # print('aafa')
                    cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255))
                    x[0],x[1]=i[0],i[1]
                    # print(i[0],i[1])
                    # print('gg')
        except:
            # print('gg')
            pass
        # image, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # try:
        #     img = cv2.drawContours(frame, contours, 3, (0, 255, 0), 3)
        # except:
        #     img = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        flag=cv2.waitKey(30)&0xFF
        cv2.imshow('',frame)
        if flag==27:
            cv2.destroyAllWindows()
            fp.close()
            break
        elif flag==ord('s'):
            if(x[0]!=0 or x[1]!=0):
                print(x)
                # y=readSTM()
                # fp.write(str(x[0])+' '+str(x[1])+' '+y[0]+' '+y[1])

if __name__ == '__main__':
    # openSerial()
    ok()
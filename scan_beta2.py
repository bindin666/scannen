import time

import cv2
import numpy as np
from time import sleep,strftime
from time import process_time as clock

import math

start = clock()


# Hardwareparameter
l = 45#in mm, die Länge der senkrechten Linie von der Kamera zur Laserlinie
h = 93.5 #in mm, der Abstand zwischen der Kamera und dem Rotationszentrum entlang der Laserrichtung
f = 981 #in Pixel, Kamerabrennweite
beta = math.atan(h/l) #Bogenmaß(Radiant), der Winkel zwischen der Mittellinie der Kamera und der vertikalen Linie
angle_one_time = 3.6 #Der Winkel jeder Drehung des Schrittmotorschritts
size9 = 1080
size16= 1920
offset_x= size9/2
# Hardwareparameter end

# Reduzieren den Rechenaufwand
camera_y = np.arange(size16)
Xi = np.arange(size9)
axis_z = 101.5*(camera_y - size16/2)/f + 0
kernel_d = np.ones((27, 5), np.uint8)
kernel_e = np.ones((10, 5), np.uint8)
lower = np.array([168, 140, 180])
upper = np.array([173, 254, 254])

# Initialisierung
cap = cv2.VideoCapture('videos.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size16)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size9)
cap.read()
# Initialisierung end


def write2file(pc,filename):
    np.savetxt(filename,pc, fmt='%.2f',newline='\n', header='')

def get_position(cap,angle):
    _, frame = cap.read()
    e2 = clock()
    print('Zeit für eine Frame ist%.2fms' % (1000 * (e2 - e1)))

    frame = cv2.bilateralFilter(frame, 5, 50, 50) # Verwenden Sie die in OpenCV integrierte Funktion bilaterFilter, um das ursprüngliche Farbbild zu entrauschen
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laser = cv2.inRange(hsv, lower, upper)  # Nehmen die Farbe, die dem Bereich entspricht
    mask = cv2.dilate(laser, kernel_d)  # die entsprichten Bereich → Expansion-Bearbeiung, und damit Gewichtsverarbeitung
    mask = cv2.erode(mask,kernel_e)
    fit = cv2.bitwise_and(gray, gray, mask=mask)  # Wählen nur das Graustufenbild aus, das die Farberkennungsbedingungen erfüllt
    thr = cv2.equalizeHist(fit)
    e3 = clock()
    print('Bildbearbeitungszeit ist%.2fms' % (1000 * (e3 - e2)))

    poi = np.array([0,0,0])
    for z in camera_y:
        if thr[:,z].sum() != 0:  # Überspringen den Fall, in dem die gesamte Linie schwarz ist

            si = np.array(Xi * thr[:,z], np.uint32)  # si ist der gewichtete Wert
            X0 = si.sum() / thr[:,z].sum()  # Berechnen den gewichteten Durchschnitt der Achsenkoordinaten des Punktes
            px = X0 - offset_x    # Versatzpixel berechnen
            r = l*math.tan(beta) - l*math.tan(beta+math.atan(px/f))  # r[y] ist der Abstand von der y-ten Höhe zum Zylinderkoordinaten
            if r>1000:
                continue
            poi_new = np.array([r*math.cos(angle/180*math.pi),r*math.sin(angle/180*math.pi),axis_z[z]])
            #print('poi_new ist：',poi_new)
            poi = np.vstack((poi,poi_new))
            #print(poi)
        else:
            continue
    e4 = clock()
    print('Die Zeit, um den Schwerpunkt zu bekommen, ist%.2fms' % (1000 * (e4 - e3)))
    return poi

# Koordinatensammlung von Punktwolken
points = np.array([0,0,0])
tem = cap.read()
tem = cap.read()
for mov in range(int(360/angle_one_time)):
    e1 = clock()
    try:
        points = np.vstack((points, get_position(cap, mov*angle_one_time)))
        # move()
        ee = clock()
        fps = 1 / (ee - e1)
        print('Die %dth Framerate beträgt ca%.2ffps\n' %(mov,fps))
    except Exception:
        pass

write2file(points,"Punktwolke_{}@{}_{}@h={}_l={}.xyz".format(strftime('%m-%d %H-%M'),size9,size16,h,l))

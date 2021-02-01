import moviepy.video.io.VideoFileClip as mpy
import numpy as np
import cv2

vid = mpy.VideoFileClip("part3.avi")

frame_count = vid.reader.nframes
video_fps = vid.fps
time_between_frames = 1/video_fps
frame1 = vid.get_frame(0)

lk_params = dict( winSize  = (25,25), maxLevel = 10, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def calc_of(old_gray, frame2, old_color,  frame1, circles, cl, colors): 
    new_color, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame2, old_color, None, **lk_params) 
    if status[0][0] == 0:
        dis = []
        for cc in circles[0]:
            dis.append( np.linalg.norm(frame1[  cc[1].astype(int), cc[0].astype(int)  ]-colors[cl]) )
        e = circles[0][ np.argmin(dis)][:2].astype(int)
        next_color = e
    else:
        next_color = new_color[0]
    next_color = next_color.reshape(-1,1,2)
    nc1 = next_color[0][0][1]
    if nc1 <= 25:
        nc1 = 25
    if nc1 >= 435:
        nc1 = 435
    next_color[0][0][1] = nc1
    nc0 = next_color[0][0][0]
    if nc0 <= 95:
        nc0 = 95
    if nc0 >= 625:
        nc0 = 625
    next_color[0][0][0] = nc0
    return next_color

def draw_frames(mask, next_color,old_color, frame1):     
    color = np.random.randint(0,255,(100,3))
    for i,(new,old) in enumerate(zip(next_color,old_color)):
        n1, n2 = new.ravel()
        o1, o2 = old.ravel()
        mask = cv2.line(mask, (n1, n2),(o1, o2), color[i].tolist(), 2)
        frame1 = cv2.circle(frame1,(n1, n2),5,color[i].tolist(),-1)
    return mask, frame1

def calc_velocity(rx, ry, gx, gy, px, py, bx, by):  
    frame1 = vid.get_frame(0)
    frame2 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
    old_gray = frame2

    old_r = np.array([[ry,rx]]).astype('float32')
    old_g = np.array([[gy,gx]]).astype('float32')
    old_p = np.array([[py,px]]).astype('float32')
    old_b = np.array([[by,bx]]).astype('float32')
    frame_r = frame1[ry, rx].astype(int)
    frame_p = frame1[py, px].astype(int)
    frame_b = frame1[by,bx].astype(int)
    frame_g = frame1[gy,gx].astype(int)

    color = np.random.randint(0,255,(100,3))
    colors = [frame_r,frame_p,frame_b,frame_g]

    vr = 0.0
    vp = 0.0
    vb = 0.0
    vg = 0.0

    mask = np.zeros_like(frame1)
    i=1
    for i in range(1, frame_count):
        frame1 = vid.get_frame(i / video_fps)
        frame2 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
        circles = cv2.HoughCircles(frame2, cv2.HOUGH_GRADIENT, 1, 15, param1=40, param2=20,minRadius=10, maxRadius=50)

        next_red = calc_of(old_gray, frame2, old_r,  frame1, circles, 0, colors)
        vr += np.linalg.norm(old_r-next_red)
        next_pink = calc_of(old_gray, frame2, old_p,   frame1, circles, 1, colors)
        vp += np.linalg.norm(old_p- next_pink)
        next_blue = calc_of(old_gray, frame2, old_b,   frame1, circles, 2, colors)
        vb += np.linalg.norm(old_b- next_blue)
        next_green = calc_of(old_gray, frame2, old_g,   frame1, circles, 3, colors)
        vg += np.linalg.norm(old_g- next_green)

        mask, frame1 = draw_frames(mask, next_red,old_r, frame1) 
        img = cv2.add(frame1,mask)
        mask, frame1 = draw_frames(mask, next_blue,old_b, frame1)     
        img = cv2.add(frame1,mask)
        mask, frame1 = draw_frames(mask, next_green,old_g, frame1)     
        img = cv2.add(frame1,mask)
        mask, frame1 = draw_frames(mask, next_pink,old_p, frame1)     
        img = cv2.add(frame1,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame2.copy()
        old_r = next_red.astype("float32")
        old_g = next_green.astype("float32")
        old_p = next_pink.astype("float32")
        old_b = next_blue.astype("float32")

    return vr, vp, vb, vg

def sortFirst(val): 
    return val[0]      

if __name__ == "__main__":
    rx, ry = 205,175
    gx, gy = 332,175
    px, py = 502,83
    bx, by = 387,323

    vr, vp, vb, vg = calc_velocity(rx, ry, gx, gy, px, py, bx, by)
    speeds = [
        [vr, "red"],
        [vg, "green"],
        [vp, "pink"],
        [vb, "blue"]
    ]
    '''
    print("Average Speeds: ") 
    print("Red:   ", speeds[0][0])
    print("Green: ", speeds[1][0])  
    print("Pink:  ", speeds[2][0]) 
    print("Blue:  ", speeds[3][0])  
    '''
    print("Balls sorted by average speed: ") 
    speeds.sort(reverse=True, key = sortFirst)  
    print(speeds) 
     
   
    
    
    
    
    
    
    
    
    
    
    
    
    






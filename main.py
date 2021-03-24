import cv2 as c
import numpy as np

#For best results, use a place with greatest color difference#
x,y,k = 200,200,-1

cap = c.VideoCapture(0)

#func
def take_inp(event, x1, y1, flag, param):
    global x, y, k
    if event == c.EVENT_LBUTTONDOWN:
        x = x1
        y = y1
        k = 1

c.namedWindow("enter_point")
c.setMouseCallback("enter_point", take_inp)


while True:

    _, inp_img = cap.read()
    inp_img = c.flip(inp_img, 1)
    gray_inp_img = c.cvtColor(inp_img, c.COLOR_BGR2GRAY)

    c.imshow("enter_point", inp_img)

    if k == 1 or c.waitKey(30) == 27:
        c.destroyAllWindows()
        break


stp = 0



old_pts = np.array([[x, y]], dtype=np.float32).reshape(-1,1,2)


mask = np.zeros_like(inp_img)

while True:
    _, new_inp_img = cap.read()
    new_inp_img = c.flip(new_inp_img, 1)
    new_gray = c.cvtColor(new_inp_img, c.COLOR_BGR2GRAY)
    new_pts,status,err = c.calcOpticalFlowPyrLK(gray_inp_img,
                         new_gray,
                         old_pts,
                         None, maxLevel=1,
                         criteria=(c.TERM_CRITERIA_EPS | c.TERM_CRITERIA_COUNT,
                                                         15, 0.08))

    for i, j in zip(old_pts, new_pts):
        x,y = j.ravel()
        a,b = i.ravel()
        if c.waitKey(2) & 0xff == ord('q'):
            stp = 1

        elif c.waitKey(2) & 0xff == ord('w'):
            stp = 0

        elif c.waitKey(2) == ord('n'):
            mask = np.zeros_like(new_inp_img)

        if stp == 0:
            mask = c.line(mask, (a,b), (x,y), (38, 0, 255), 6)

        c.circle(new_inp_img, (x,y), 6, (38, 0, 255), -1)

    new_inp_img = c.addWeighted(mask, 0.3, new_inp_img, 0.7, 0)
    c.putText(mask, "'q' to gap 'w' - start 'n' - clear 'esc' - exit", (10,50),
                c.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    c.imshow("Camera", new_inp_img)
    c.imshow("Result", mask)


    gray_inp_img = new_gray.copy()
    old_pts = new_pts.reshape(-1,1,2)

    if c.waitKey(1) & 0xff == 27:
        break

c.destroyAllWindows()
cap.release()

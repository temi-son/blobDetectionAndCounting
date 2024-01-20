# importing libraries and dependencies
import cv2 as cv
import numpy as np

# declaration and initilisation of variables
dropletVideo = cv.VideoCapture('VideoCW2.mp4')
prevWrap = None
def dist(x1, y1, x2, y2): return (x1-x2)**2*(y1-y2)**2


count = 0   # used for counting the successfully formed droplets
flag = 1  # used as a starting point for counting the successfully formed droplets
position = (460,40)

# main loop
while True:

    # reading the input video input and decleration
    ret, frame = dropletVideo.read()

    if not ret:
        break

    # marks the point where counting is done
    cv.line(frame, (448, 0), (448, 200), (0, 0, 255), 2)

# USING BLOB DETECTION TO DETECT INNER DROPLET AND CENTER OF MASS
    # Change the frame to HSV
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # thresholds the hsv frame to detect just brown pixels
    thresholdFrame = cv.inRange(hsvFrame, (10, 100, 20), (30, 255, 200))

    # filtes the threshold frame to clean up edges
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
    thresholdFrame = cv.erode(thresholdFrame, kernel, iterations=1)
    thresholdFrame = cv.dilate(thresholdFrame, kernel, iterations=1)

    # blob detection and center of blob
    contours, _ = cv.findContours(
        thresholdFrame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # contour drawn around the inner droplet
        cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        M = cv.moments(cnt)
        if M['m00'] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

            # circle drawn around the center of mass
            cv.circle(frame, (x, y), 2, (255, 0, 50), 2)

            # counting successfully formed droplets
            if flag == 1:
                if x < 440:
                    count = count + 1
            elif x == 440:
                count = count + 1  # adds 1 to count whenever the droplet passes the point
    flag = 0  # set the flag to 0

    # displays the number of successfully formed droplets on the frame
    cv.putText(frame, "Counting blobs... " + str(count), position,
               cv.FONT_HERSHEY_SIMPLEX, 1, (115, 50, 255), 2)


# USING HOUGH CIRCLE TO DETECT OUTER WRAP
    # conversion to gray scale
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # noise reduction
    blurFrame = cv.medianBlur(grayFrame, 1)

    # performing HoughCirle to detect the outer wrap
    outerWrap = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT,
                                1, 5, param1=200, param2=15, minRadius=30, maxRadius=30)

    # detecting the outer wrap
    if outerWrap is not None:
        outerWrap = np.uint16(np.around(outerWrap))
        initialWrap = None
        for i in outerWrap[0, :]:
            if initialWrap is None:
                initialWrap = i
            if prevWrap is not None:
                if dist(initialWrap[0], initialWrap[1], prevWrap[0], prevWrap[1]) <= dist(i[0], i[1], 
                                                                                          prevWrap[0], prevWrap[1]):
                    initialWrap = i
        # drawing a circle around the outer wrap
        cv.circle(frame, (initialWrap[0], initialWrap[1]),
                  initialWrap[2], (255, 0, 255), 3)
        prevWrap = initialWrap

    # displays the  threshold frame
    cv.imshow("Threshold Frame", thresholdFrame)
    # displays the frame, showing the center of mass, outer wrap, inner droplet and count
    cv.imshow("Main Frame", frame)

    # closing the frame
    if cv.waitKey(1) & 0xFF == 27:
        break

dropletVideo.release()
cv.destroyAllWindows()

import cv2
import numpy as np
import utlis

###################################################
width = 700
height = 960
pathImage = r"D:\Dropbox\python\openCV\OMR\Capture4.PNG"
#pathImage = r"D:\Dropbox\python\openCV\OMR\5.jpg"
qNum = 12 #number of questions
nChoices = 4 #number of choices
maxGrade = 100
answers = [0,3,3,3,3,0,2,2,3,3,2,3]
###################################################
img = cv2.imread(pathImage)

# Pre Processing
img = cv2.resize(img,(width,height))
imgFinal = img.copy()
imgContours = img.copy()
imgBiggestContour = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

#Finding all contours
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours,-1,(0,255,0),10)

# Find Rectangle Contours
rectCon = utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
#print(biggestContour)
gradePoints = utlis.getCornerPoints(rectCon[1])
#print(gradePoints)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContour,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContour,gradePoints,-1,(255,0,0),20)
    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix,(width,height))

    ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
    ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
    matrixG = cv2.getPerspectiveTransform(ptsG1,ptsG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG,(325,150))


# Apply threshild 
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh =cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

    
    boxes = utlis.splitBoxes(imgThresh,qNum,nChoices)
    cv2.imshow("test",boxes[0])

# Find the boxes with heighest non zero pixels
    #print(cv2.countNonZero(boxes[0]), cv2.countNonZero(boxes[1]))
    myPixelsVal = np.zeros((qNum,nChoices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelsVal[countR][countC] = totalPixels
        countC += 1
        if (countC == nChoices): countR +=1 ; countC = 0
    #print(myPixelsVal)

# find index values of the marking in each raw
    myIndex = []
    for x in range (0,qNum):
        arr = myPixelsVal[x]
        #print('arr',arr)
        myIndexVal = np.where(arr == np.max(arr))
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    #print(myIndex)

# Grading
    grading = []
    for x in range (0,qNum):
        if answers[x] == myIndex[x]:
            grading.append(1)
        else: grading.append(0)
    #print(grading)


# find the final score
    score = (sum(grading)/qNum)* maxGrade
    #print(score)


# Displaying answers and score
    imgResults = imgWarpColored.copy()
    imgResults = utlis.showAnswers(imgResults,myIndex,grading,answers,qNum,nChoices)

# Mask and combine the answers over the original image
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utlis.showAnswers(imgRawDrawing,myIndex,grading,answers,qNum,nChoices)
    invmatrix = cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invmatrix,(width,height))
    


# Mask and combine the grade over the original image
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade,str(round(score,1))+"%",(30,100),cv2.FONT_HERSHEY_SIMPLEX,3,(250,250,250),5)
    invMatrixG = cv2.getPerspectiveTransform(ptsG2,ptsG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade,invMatrixG,(width,height))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
    imgFinal = cv2.addWeighted(imgFinal,0.8, imgInvGradeDisplay, -1,0)


cv2.imshow("grade",imgRawGrade)
imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray,imgBlur, imgCanny],
[imgContours, imgBiggestContour,imgWarpColored,imgThresh],
[imgResults,imgRawDrawing,imgInvWarp,imgFinal])
imgStacked = utlis.stackImages(imageArray,0.3)

cv2.imshow('final',imgFinal)
cv2.imshow("Stacked Images", imgStacked)
cv2.moveWindow("Stacked Images",0,0)
cv2.waitKey(0)
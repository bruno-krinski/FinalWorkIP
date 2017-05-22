import cv2

nep = cv2.imread("nep.png", 0)
cv2.imshow("nep",nep)
cv2.waitKey(0)
cv2.destroyAllWindows()

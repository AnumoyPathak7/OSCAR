import cv2,numpy as np
import pytesseract
def extract(path):
    im=cv2.imread(str(path))
    if im is None: raise FileNotFoundError(path)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); g=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(g)
    g=cv2.bilateralFilter(g,9,75,75); g=cv2.bilateralFilter(g,9,75,75)
    g=cv2.filter2D(g,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    g=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    g=cv2.morphologyEx(g,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    try:
        text=pytesseract.image_to_string(g,config="--oem 3 --psm 11").strip()
        data=pytesseract.image_to_data(g,config="--oem 3 --psm 11",output_type=pytesseract.Output.DICT)
        boxes=[{"text":t.strip(),"left":int(data["left"][i]),"top":int(data["top"][i]),"width":int(data["width"][i]),"height":int(data["height"][i])} for i,t in enumerate(data["text"]) if t.strip()]
        return text,boxes
    except pytesseract.TesseractNotFoundError: return "",[{"error":"Tesseract is not installed or not on PATH"}]

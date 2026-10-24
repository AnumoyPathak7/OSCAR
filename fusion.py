def adaptive_fusion(image_conf,text_conf,occlusion=0.0,image_weight=1.0,text_weight=1.0):
    o=max(0.,min(1.,float(occlusion))); ci=max(0.,min(1.,float(image_conf))); ct=max(0.,min(1.,float(text_conf)))
    wi=image_weight*(1-.5*o); wt=text_weight*(.25+.75*o); d=max(wi+wt,1e-8)
    return {"fused_confidence":(wi*ci+wt*ct)/d,"image_weight":wi/d,"text_weight":wt/d}
def estimate_occlusion(image):
    import cv2,numpy as np
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    ratio=float(np.mean(cv2.Canny(g,60,150)>0))
    return max(0.,min(1.,(ratio-.04)/.25))

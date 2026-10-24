import json,torch,cv2
from PIL import Image
from torchvision import transforms
from .train import device_of,load_model
from .model import mc_dropout
from .ocr import extract
from .fusion import adaptive_fusion,estimate_occlusion
def predict(image_path,checkpoint,mc_samples=10,tta=False,device="auto"):
    dev=device_of(device); m,c=load_model(checkpoint,dev); size=c["image_size"]
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    with Image.open(image_path) as im: x=tf(im.convert("RGB")).unsqueeze(0).to(dev)
    m.eval(); samples=[]
    if mc_samples>1:
        mc_dropout(m)
        with torch.no_grad():
            for _ in range(mc_samples): samples.append(torch.softmax(m(x),1))
        probs=torch.stack(samples).mean(0)[0]; var=float(torch.stack(samples).var(0).mean().cpu()); m.eval()
    else:
        with torch.no_grad(): probs=torch.softmax(m(x),1)[0]
        var=0.
    if tta:
        with torch.no_grad(): probs=(probs+torch.softmax(m(torch.flip(x,[3])),1)[0])/2
    conf,idx=probs.max(0); label=c["labels"][int(idx)]; raw=cv2.imread(str(image_path)); occ=estimate_occlusion(raw) if raw is not None else 0.
    text,boxes=extract(image_path); tc=min(.85,.35+.08*len(text.split())) if text else 0.
    fusion=adaptive_fusion(float(conf.cpu()),tc,occ)
    result={"predicted_class":label,"image_confidence":round(float(conf.cpu()),4),"mc_dropout_variance":round(var,6),"ocr_text":text,"ocr_boxes":boxes,"estimated_occlusion":round(occ,3),"fusion":{k:round(v,4) for k,v in fusion.items()},"note":"OCR confidence is a heuristic, not a trained text-to-class classifier; fused confidence is experimental."}
    print(json.dumps(result,indent=2,ensure_ascii=False)); return result

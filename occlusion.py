from pathlib import Path
import random,csv,cv2,numpy as np
EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
def draw(image):
    h,w=image.shape[:2]; out=image.copy(); c=tuple(random.randint(0,255) for _ in range(3)); c=c+((255,) if out.ndim==3 and out.shape[2]==4 else ())
    shape=random.choice(["rectangle","circle","polygon"])
    if shape=="rectangle":
        x1,x2=sorted([random.randrange(w),random.randrange(w)]); y1,y2=sorted([random.randrange(h),random.randrange(h)])
        cv2.rectangle(out,(x1,y1),(max(x1+1,x2),max(y1+1,y2)),c,-1)
    elif shape=="circle": cv2.circle(out,(random.randrange(w),random.randrange(h)),random.randint(3,max(4,min(w,h)//3)),c,-1)
    else: cv2.fillPoly(out,[np.array([[random.randrange(w),random.randrange(h)] for _ in range(random.randint(3,6))],np.int32)],c)
    return out
def generate(input_dir,output_dir,copies=5):
    src,dst=Path(input_dir),Path(output_dir); count=0
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS: continue
        im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if im is None: continue
        if im.ndim==2: im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGRA)
        elif im.shape[2]==3: im=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
        rel=p.relative_to(src); target=dst/rel.parent; target.mkdir(parents=True,exist_ok=True)
        for i in range(1,copies+1):
            cv2.imwrite(str(target/f"{p.stem}_occ{i}.png"),draw(im)); count+=1
    rows=[{"filepath":str(p.resolve()),"label":p.parent.name} for p in dst.rglob("*.png")]
    if rows:
        with (dst/"labeled_dataset.csv").open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=["filepath","label"]); w.writeheader(); w.writerows(rows)
    return count
def validate(input_dir,output_dir,copies=5):
    src,dst=Path(input_dir),Path(output_dir); missing=[]
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            for i in range(1,copies+1):
                q=dst/p.relative_to(src).parent/f"{p.stem}_occ{i}.png"
                if not q.exists(): missing.append(str(q))
    print("Missing",len(missing),"files. Examples:",missing[:10]) if missing else print("All occluded files generated successfully.")

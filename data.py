from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
def make_csv(input_dir,output_csv):
    root=Path(input_dir).resolve(); rows=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            # class folder is immediate parent; use a class-folder dataset layout
            rows.append({"filepath":str(p.resolve()),"label":p.parent.name})
    if not rows: raise ValueError("No images found. Use class subfolders.")
    out=Path(output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False); return len(rows)
def validate_csv(path):
    df=pd.read_csv(path)
    if not {"filepath","label"}.issubset(df.columns): raise ValueError("CSV must have filepath,label columns")
    if df.empty or df.label.isna().any(): raise ValueError("CSV is empty or contains missing labels")
    base=Path(path).resolve().parent
    bad=[str((base/Path(v)).resolve()) for v in df.filepath.astype(str) if not (Path(v).is_absolute() and Path(v).exists()) and not (base/Path(v)).exists()]
    if bad: raise FileNotFoundError(f"Missing image files, examples: {bad[:5]}")
    return df
def transform(size=128,train=False):
    ops=[transforms.Resize((size,size))]
    if train: ops += [transforms.RandomResizedCrop(size,scale=(.75,1)),transforms.RandomRotation(12),transforms.ColorJitter(.25,.25,.2,.04),transforms.RandomHorizontalFlip()]
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
class SignDataset(Dataset):
    def __init__(self,csv,label_map=None,size=128,train=False):
        self.path=Path(csv).resolve(); self.df=validate_csv(csv); self.base=self.path.parent
        labels=sorted(self.df.label.astype(str).unique()); self.label_map=label_map or {v:i for i,v in enumerate(labels)}
        if set(labels)-set(self.label_map): raise ValueError("CSV contains labels absent from training label map")
        self.tf=transform(size,train)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; p=Path(str(r.filepath)); p=p if p.is_absolute() else self.base/p
        with Image.open(p) as im: x=self.tf(im.convert("RGB"))
        return x,self.label_map[str(r.label)]

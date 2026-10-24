from pathlib import Path
import json,copy,random,numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from tqdm import tqdm
from .data import SignDataset,validate_csv
from .model import OSCARNet
def device_of(v): return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if v=="auto" else v)
@torch.no_grad()
def ema_update(t,s,decay=.995):
    for a,b in zip(t.parameters(),s.parameters()): a.data.mul_(decay).add_(b.data,alpha=1-decay)
    for a,b in zip(t.buffers(),s.buffers()):
        if a.dtype.is_floating_point: a.data.mul_(decay).add_(b.data,alpha=1-decay)
        else: a.data.copy_(b.data)
def load_model(path,device):
    c=torch.load(path,map_location=device,weights_only=False); m=OSCARNet(c["num_classes"],False).to(device); m.load_state_dict(c["model_state"]); m.eval(); return m,c
def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=device_of(a.device); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    df=validate_csv(a.csv); tr,va=train_test_split(df,test_size=a.val_fraction,random_state=a.seed,stratify=df.label)
    tr.to_csv(out/"train.csv",index=False); va.to_csv(out/"val.csv",index=False)
    labels=sorted(df.label.astype(str).unique()); lm={v:i for i,v in enumerate(labels)}; (out/"labels.json").write_text(json.dumps(lm,indent=2))
    dl=DataLoader(SignDataset(out/"train.csv",lm,a.image_size,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
    vl=DataLoader(SignDataset(out/"val.csv",lm,a.image_size,False),batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    s=OSCARNet(len(labels),True).to(dev); t=copy.deepcopy(s).to(dev).eval()
    for p in t.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(s.parameters(),lr=a.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(label_smoothing=.05); scaler=torch.amp.GradScaler("cuda",enabled=dev.type=="cuda"); best=-1
    for ep in range(1,a.epochs+1):
        s.train(); losses=[]
        for x,y in tqdm(dl,desc=f"Epoch {ep}/{a.epochs}"):
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=dev.type=="cuda"):
                so=s(x)
                with torch.no_grad(): to=t(x)
                loss=ce(so,y)+.3*nn.functional.kl_div(nn.functional.log_softmax(so/2,1),nn.functional.softmax(to/2,1),reduction="batchmean")*4
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); ema_update(t,s); losses.append(loss.item())
        s.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,y in vl: ps.extend(s(x.to(dev)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
        acc=accuracy_score(ys,ps); f1=f1_score(ys,ps,average="macro",zero_division=0)
        print(f"epoch {ep}: loss={np.mean(losses):.4f} val_accuracy={acc:.4f} macro_f1={f1:.4f}")
        c={"model_state":s.state_dict(),"teacher_state":t.state_dict(),"num_classes":len(labels),"labels":labels,"label_to_idx":lm,"image_size":a.image_size,"epoch":ep,"val_accuracy":acc}
        torch.save(c,out/"last.pt")
        if acc>best: best=acc; torch.save(c,out/"best.pt")
    print("Best validation accuracy:",best,"checkpoint:",out/"best.pt")
def evaluate(a):
    dev=device_of(a.device); m,c=load_model(a.checkpoint,dev); ds=SignDataset(a.csv,c["label_to_idx"],c["image_size"],False); dl=DataLoader(ds,batch_size=a.batch_size)
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl: ps.extend(m(x.to(dev)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
    print("Accuracy:",accuracy_score(ys,ps)); print("Macro F1:",f1_score(ys,ps,average="macro",zero_division=0)); print(classification_report(ys,ps,labels=list(range(len(c["labels"]))),target_names=c["labels"],zero_division=0))

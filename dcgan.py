from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,utils
from tqdm import tqdm
from .train import device_of
class Generator(nn.Module):
    def __init__(self,zdim=100,base=64):
        super().__init__(); self.net=nn.Sequential(nn.ConvTranspose2d(zdim,base*8,4,1,0,bias=False),nn.BatchNorm2d(base*8),nn.ReLU(True),nn.ConvTranspose2d(base*8,base*4,4,2,1,bias=False),nn.BatchNorm2d(base*4),nn.ReLU(True),nn.ConvTranspose2d(base*4,base*2,4,2,1,bias=False),nn.BatchNorm2d(base*2),nn.ReLU(True),nn.ConvTranspose2d(base*2,base,4,2,1,bias=False),nn.BatchNorm2d(base),nn.ReLU(True),nn.ConvTranspose2d(base,3,4,2,1,bias=False),nn.Tanh())
    def forward(self,z): return self.net(z)
class Discriminator(nn.Module):
    def __init__(self,base=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,base,4,2,1,bias=False),nn.LeakyReLU(.2,True),nn.Conv2d(base,base*2,4,2,1,bias=False),nn.BatchNorm2d(base*2),nn.LeakyReLU(.2,True),nn.Conv2d(base*2,base*4,4,2,1,bias=False),nn.BatchNorm2d(base*4),nn.LeakyReLU(.2,True),nn.Conv2d(base*4,base*8,4,2,1,bias=False),nn.BatchNorm2d(base*8),nn.LeakyReLU(.2,True),nn.Conv2d(base*8,1,4,1,0,bias=False),nn.Sigmoid())
    def forward(self,x): return self.net(x).view(-1)
def train_gan(a):
    dev=device_of(a.device); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tf=transforms.Compose([transforms.Resize(a.image_size),transforms.CenterCrop(a.image_size),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    ds=datasets.ImageFolder(a.input,transform=tf)
    if not len(ds): raise ValueError("DCGAN expects images in class subfolders")
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=True,num_workers=2)
    g=Generator(a.latent_dim).to(dev); d=Discriminator().to(dev); bce=nn.BCELoss()
    og=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999)); fixed=torch.randn(64,a.latent_dim,1,1,device=dev)
    for ep in range(1,a.epochs+1):
        for real,_ in tqdm(dl,desc=f"DCGAN {ep}/{a.epochs}"):
            real=real.to(dev); n=real.size(0); z=torch.randn(n,a.latent_dim,1,1,device=dev); fake=g(z)
            od.zero_grad(set_to_none=True); ld=bce(d(real),torch.ones(n,device=dev))+bce(d(fake.detach()),torch.zeros(n,device=dev)); ld.backward(); od.step()
            og.zero_grad(set_to_none=True); lg=bce(d(fake),torch.ones(n,device=dev)); lg.backward(); og.step()
        with torch.no_grad(): utils.save_image(g(fixed).cpu(),out/f"samples_{ep:03d}.png",normalize=True,nrow=8)
        torch.save({"generator":g.state_dict(),"latent_dim":a.latent_dim},out/"dcgan.pt"); print(ep,"D",ld.item(),"G",lg.item())
def generate_gan(checkpoint,output,count=100,device="auto"):
    dev=device_of(device); out=Path(output); out.mkdir(parents=True,exist_ok=True); c=torch.load(checkpoint,map_location=dev,weights_only=False)
    g=Generator(c["latent_dim"]).to(dev); g.load_state_dict(c["generator"]); g.eval()
    with torch.no_grad():
        for i in range(count):
            utils.save_image(g(torch.randn(1,c["latent_dim"],1,1,device=dev))[0],out/f"synthetic_{i:05d}.png",normalize=True)
    print("Generated",count,"images; review them before use.")

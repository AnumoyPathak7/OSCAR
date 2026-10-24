import torch
from torch import nn
def fit_temperature(logits,labels,max_iter=50):
    t=nn.Parameter(torch.ones(1,device=logits.device)*1.5); opt=torch.optim.LBFGS([t],lr=.01,max_iter=max_iter); loss_fn=nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss=loss_fn(logits/t.clamp(min=.05),labels); loss.backward(); return loss
    opt.step(closure); return float(t.detach().clamp(min=.05).cpu())

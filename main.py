import argparse
from oscar.data import make_csv, validate_csv
from oscar.occlusion import generate, validate
from oscar.train import train, evaluate
from oscar.inference import predict
from oscar.dcgan import train_gan, generate_gan

def main():
    p=argparse.ArgumentParser(description="OSCAR traffic-sign framework")
    s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("make-csv"); a.add_argument("--input",required=True); a.add_argument("--output",required=True)
    a=s.add_parser("occlude"); a.add_argument("--input",required=True); a.add_argument("--output",required=True); a.add_argument("--copies",type=int,default=5)
    a=s.add_parser("validate-occlusions"); a.add_argument("--input",required=True); a.add_argument("--output",required=True); a.add_argument("--copies",type=int,default=5)
    a=s.add_parser("train"); a.add_argument("--csv",required=True); a.add_argument("--output",required=True); a.add_argument("--epochs",type=int,default=30); a.add_argument("--batch-size",type=int,default=32); a.add_argument("--image-size",type=int,default=128); a.add_argument("--lr",type=float,default=3e-4); a.add_argument("--workers",type=int,default=2); a.add_argument("--val-fraction",type=float,default=.15); a.add_argument("--seed",type=int,default=42); a.add_argument("--device",default="auto")
    a=s.add_parser("evaluate"); a.add_argument("--csv",required=True); a.add_argument("--checkpoint",required=True); a.add_argument("--batch-size",type=int,default=32); a.add_argument("--device",default="auto")
    a=s.add_parser("predict"); a.add_argument("--image",required=True); a.add_argument("--checkpoint",required=True); a.add_argument("--mc-samples",type=int,default=10); a.add_argument("--tta",action="store_true"); a.add_argument("--device",default="auto")
    a=s.add_parser("train-dcgan"); a.add_argument("--input",required=True); a.add_argument("--output",required=True); a.add_argument("--epochs",type=int,default=50); a.add_argument("--batch-size",type=int,default=64); a.add_argument("--image-size",type=int,default=64); a.add_argument("--latent-dim",type=int,default=100); a.add_argument("--device",default="auto")
    a=s.add_parser("generate-dcgan"); a.add_argument("--checkpoint",required=True); a.add_argument("--output",required=True); a.add_argument("--count",type=int,default=100); a.add_argument("--device",default="auto")
    x=p.parse_args()
    if x.cmd=="make-csv": print("Rows:",make_csv(x.input,x.output))
    elif x.cmd=="occlude": print("Generated:",generate(x.input,x.output,x.copies))
    elif x.cmd=="validate-occlusions": validate(x.input,x.output,x.copies)
    elif x.cmd=="train": validate_csv(x.csv); train(x)
    elif x.cmd=="evaluate": evaluate(x)
    elif x.cmd=="predict": predict(x.image,x.checkpoint,x.mc_samples,x.tta,x.device)
    elif x.cmd=="train-dcgan": train_gan(x)
    elif x.cmd=="generate-dcgan": generate_gan(x.checkpoint,x.output,x.count,x.device)
if __name__=="__main__": main()

"""Word-level autoregressive comparison for Assignment 2.

Run: python main.py --epochs 8 --max-tokens 30000
Use --max-tokens 0 to train on the full corpus. Results are saved under results/.
"""
import argparse, collections, json, math, os, random, re, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import SimpleRNNModel, BiRNNModel, LSTMModel, GRUModel

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)

def tokenize(text): return TOKEN_RE.findall(text.lower())

class Windows(Dataset):
    def __init__(self, ids, length, stride=5):
        self.ids, self.length, self.stride = ids, length, stride
    def __len__(self): return max(0, (len(self.ids)-self.length-1)//self.stride+1)
    def __getitem__(self, i):
        i *= self.stride
        a = self.ids[i:i+self.length]
        return torch.tensor(a), torch.tensor(self.ids[i+self.length])

def metrics(model, loader, device):
    model.eval(); loss_sum=correct=count=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits,_=model(x); logits=logits[:,-1,:]
            loss_sum += nn.functional.cross_entropy(logits,y,reduction='sum').item()
            correct += (logits.argmax(-1)==y).sum().item(); count += y.numel()
    loss=loss_sum/max(count,1)
    return loss, math.exp(min(loss, 80)), correct/max(count,1)

def generate(model, seed, stoi, itos, length, temp, device):
    model.eval(); words=tokenize(seed); words=[w if w in stoi else UNK for w in words]
    with torch.no_grad():
        for _ in range(length):
            ids=[stoi.get(w,stoi[UNK]) for w in words[-40:]]
            logits,_=model(torch.tensor([ids],device=device)); p=torch.softmax(logits[0,-1]/temp,dim=-1)
            p[stoi[PAD]]=0
            nxt=torch.multinomial(p,1).item(); words.append(itos[nxt])
    # readable punctuation spacing
    out=''
    for w in words:
        if w in '.,!?;:)%]}' or w in ('\'s','\'t','\'re','\'ve','\'ll','\'d'):
            out=out.rstrip()+w+' '
        elif w in ('(', '[', '{'):
            out+=w
        else: out+=w+' '
    return out.strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--max-tokens',type=int,default=30000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--batch-size',type=int,default=64)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    raw=open('tinyshakespeare.txt',encoding='utf-8').read(); all_words=tokenize(raw)
    if args.max_tokens: all_words=all_words[:args.max_tokens]
    n=len(all_words); a=int(.8*n); b=int(.9*n); train_words=all_words[:a]; val_words=all_words[a:b]; test_words=all_words[b:]
    counts=collections.Counter(train_words); vocab=[PAD,UNK]+[w for w,_ in counts.most_common(5998) if w not in (PAD,UNK)]
    stoi={w:i for i,w in enumerate(vocab)}; itos=vocab
    encode=lambda ws:[stoi.get(w,1) for w in ws]
    train=Windows(encode(train_words),20); val=Windows(encode(val_words),20); test=Windows(encode(test_words),20)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders=[DataLoader(d,batch_size=args.batch_size,shuffle=(i==0),drop_last=False) for i,d in enumerate((train,val,test))]
    seeds=["To be or not to be", "my lord what is thy name", "the king and the lord"]
    missing=[w for seed in seeds for w in tokenize(seed) if w not in stoi]
    if missing: raise ValueError(f"Seed words missing from training vocabulary: {sorted(set(missing))}")
    configs=[('SimpleRNN',SimpleRNNModel),('BiRNN',BiRNNModel),('LSTM',LSTMModel),('GRU',GRUModel)]
    os.makedirs('results',exist_ok=True); result={'dataset':{'name':'Tiny Shakespeare','source':'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt','tokens_used':n,'train_tokens':len(train_words),'validation_tokens':len(val_words),'test_tokens':len(test_words),'vocabulary_size':len(vocab),'tokenization':'lowercased words and punctuation; max 6,000 vocabulary items, remaining words mapped to <unk>'},'settings':{'seed':args.seed,'epochs':args.epochs,'sequence_length':20,'window_stride':5,'batch_size':args.batch_size,'optimizer':'Adam','learning_rate':0.001,'device':str(device)},'models':{}}
    for name, cls in configs:
        torch.manual_seed(args.seed)
        model=cls(len(vocab),embed_size=64,hidden_size=64,num_layers=1,dropout=0.1).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=.001); history=[]; start=time.time()
        for epoch in range(args.epochs):
            model.train(); total=cnt=0; t0=time.time()
            for x,y in loaders[0]:
                x,y=x.to(device),y.to(device); opt.zero_grad(); logits,_=model(x); logits=logits[:,-1,:]
                loss=nn.functional.cross_entropy(logits,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                total+=loss.item()*y.numel(); cnt+=y.numel()
            vl,vp,va=metrics(model,loaders[1],device)
            history.append({'epoch':epoch+1,'train_loss':total/cnt,'val_loss':vl,'val_perplexity':vp,'val_accuracy':va,'epoch_seconds':time.time()-t0})
            print(f"{name} epoch {epoch+1}/{args.epochs}: val loss {vl:.4f}, ppl {vp:.2f}, acc {va:.3f}",flush=True)
        tl,tp,ta=metrics(model,loaders[2],device)
        samples={str(t):{s:generate(model,s,stoi,itos,50,t,device) for s in seeds} for t in (.5,1.0,1.5)}
        torch.save(model.state_dict(),f'results/{name}.pt')
        result['models'][name]={'history':history,'test_loss':tl,'test_perplexity':tp,'test_accuracy':ta,'training_seconds':sum(row['epoch_seconds'] for row in history),'end_to_end_seconds':time.time()-start,'samples':samples}
        with open('results/results.json','w',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    with open('results/results.json','w',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print('Saved results/results.json; see README for interpretation guidance.')
if __name__=='__main__': main()

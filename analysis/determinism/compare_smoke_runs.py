import json,sys,os,numpy as np
def load(d):
    r=json.load(open(d+'/results.json'))
    tl={(x['round'],k):v['train_loss'] for x in r['rounds'] for k,v in x['fit']['clients'].items()}
    return r,tl
def cmp(a,b):
    ra,ta=load(a); rb,tb=load(b)
    print('\n#### %s  vs  %s'%(a,b))
    for key in sorted(ta):
        x,y=ta[key],tb.get(key); print('  train_loss r%d %s: %r vs %r -> %s'%(key[0],key[1],x,y,'EQUAL' if x==y else 'DIFF %.3g'%abs(x-y)))
    va,vb=ra['model_selection']['val_loss_by_round'],rb['model_selection']['val_loss_by_round']
    for k in va: print('  agg val loss r%s: %r vs %r -> %s'%(k,va[k],vb[k],'EQUAL' if va[k]==vb[k] else 'DIFF'))
    for f in sorted(os.listdir(a+'/predictions')):
        if not f.endswith('.npz'): continue
        A=np.load(a+'/predictions/'+f); B=np.load(b+'/predictions/'+f)
        same=all(np.array_equal(A[k],B[k]) for k in A.files)
        d=np.abs(A['logit_margin']-B['logit_margin'])
        print('  %s: %s'%(f,'bitwise' if same else 'DIFF max|dm|=%.3g n=%d/%d flips=%d'%(d.max(),(d>0).sum(),d.size,((A['logit_margin']>0)!=(B['logit_margin']>0)).sum())))
for a,b in zip(sys.argv[1::2],sys.argv[2::2]): cmp(a,b)

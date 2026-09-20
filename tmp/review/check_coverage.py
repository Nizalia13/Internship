import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tmp/review/deps'))
sys.path.insert(0, str(ROOT/'Conformal Library'))
import numpy as np
import pandas as pd
from conformal_envelopes import ConformalSetModel

scores = pd.read_parquet(ROOT/'Data/phage_host_scores.parquet')
meta = pd.read_csv(ROOT/'Data/metadata.csv', usecols=['accession','host','host_type','split'])
group = meta.groupby('accession')
print('metadata conflicts', int(group[['host','host_type','split']].nunique().gt(1).any(axis=1).sum()), flush=True)
df = scores.join(group[['host','host_type','split']].first()).rename_axis('ID').reset_index()
cols = scores.columns.tolist()
hosts = sorted({c.split('_',1)[0] for c in cols})
df = df[df.host.isin(hosts)].copy()
train = df[~df.split.isin([0,1])].copy()
val = df[df.split.eq(1)].copy()
counts = train.host.value_counts()
hosts = [h for h in hosts if counts.get(h,0)>=3]
train = train[train.host.isin(hosts)]
mapping = {h:[c for c in cols if c.split('_',1)[0]==h] for h in hosts}
gp = pd.read_csv(ROOT/'Codes/gram_type_predictions.csv').set_index('accession')
for c in ['pred_gramneg','pred_grampos']:
 gp[c] = gp[c].astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
hostgram = df[['host','host_type']].drop_duplicates().set_index('host').host_type.to_dict()
candidates = {}
for sid in val.ID:
 if sid not in gp.index: candidates[sid]=hosts;continue
 row=gp.loc[sid]
 candidates[sid]=[h for h in hosts if (hostgram[h]=='gram-neg' and row.pred_gramneg) or (hostgram[h]=='gram-pos' and row.pred_grampos)] or hosts
allowed=np.array([h in candidates[s] for s,h in zip(val.ID,val.host)])
print('train/validation',len(train),len(val),'candidate true-host retention',allowed.mean(),flush=True)
results={}
for method,params in [('collapsed',{}),('radial',{'n_directions':100,'smoothing':4.0}),('strip',{'n_bins':10,'min_samples':3})]:
 t=time.time()
 model=ConformalSetModel(method=method,random_state=42,force_nonempty=False,**params).fit(train,id_col='ID',label_col='host',score_cols=cols,label_to_columns=mapping)
 pred=model.predict(val)
 covered=np.array([h in s for h,s in zip(val.host,pred.prediction_set)])
 filtered=[ [h for h in ss if h in candidates[sid]] for sid,ss in zip(val.ID,pred.prediction_set)]
 detail=[]
 for h in hosts:
  ix=(val.host.to_numpy()==h)
  info=model.envelopes_[h];e=info['envelope']
  nc=1-train.loc[train.host.eq(h),info['columns']].to_numpy(float)
  s2=nc[info['idx_s2']]
  if method=='strip':
   from conformal_envelopes.envelopes.strip import strip_is_in_region
   cin=strip_is_in_region(s2,e)
  elif method=='radial':
   from conformal_envelopes.envelopes.radial import radial_is_in_region
   cin=radial_is_in_region(s2,e)
  else:
   from conformal_envelopes.envelopes.collapsed import collapsed_is_in_region
   cin=collapsed_is_in_region(s2,e)
  detail.append({'host':h,'n_cal':len(s2),'n_val':int(ix.sum()),'cal_coverage':float(cin.mean()),'val_coverage':float(covered[ix].mean()) if ix.any() else None,'t_hat':float(e['t_hat'])})
 summary={'unfiltered_coverage':float(covered.mean()),'filtered_coverage':float((covered & allowed).mean()),'filter_only_losses':int((covered & ~allowed).sum()),'unfiltered_size':float(pred.set_size.mean()),'filtered_size':float(np.mean([len(s) for s in filtered])),'seconds':time.time()-t,'classes':detail}
 results[method]=summary
 print(method,json.dumps({k:v for k,v in summary.items() if k!='classes'}),flush=True)
 print('largest classes', sorted(detail,key=lambda r:-r['n_val'])[:6],flush=True)
 (ROOT/'tmp/review/coverage_results.json').write_text(json.dumps(results,indent=2))

"""Regenerate every target's prediction and five-seed ranking figures."""
import argparse,csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);a=parser.parse_args()
    for panel in ['four_target','k4dd']:
        with (a.output_root/'predictions'/panel/'unique_compound_five_seed_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        for index,target in enumerate(sorted({r['uniprot_id'] for r in rows}),1):
            group=[r for r in rows if r['uniprot_id']==target]
            y=np.array([float(r['observed_pkoff']) for r in group])
            pred=np.array([[float(r[f'predicted_pkoff_seed_{s}']) for s in [42,142,242,342,442]] for r in group])
            if not np.isfinite(pred).all():raise RuntimeError('Nonfinite panel predictions')
            mean=pred.mean(1);sd=pred.std(1,ddof=1)
            fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.8))
            ax.errorbar(y,mean,yerr=sd,fmt='o',ms=4,color='#303030',ecolor='#aaaaaa',capsize=2)
            low=min(y.min(),mean.min())-.1;high=max(y.max(),mean.max())+.1
            ax.plot([low,high],[low,high],ls='--',color='gray',lw=1)
            ax.set(xlabel='Observed pKoff',ylabel='Predicted pKoff (mean +/- seed SD)',title='Absolute prediction')
            order=np.argsort(-y,kind='stable')
            ranks=rankdata(-pred,axis=0,method='average')
            image=bx.imshow(ranks[order],aspect='auto',cmap='Greys_r',vmin=1,vmax=max(2,len(group)))
            bx.set_xticks(range(5));bx.set_xticklabels([42,142,242,342,442])
            bx.set(xlabel='Refit seed',ylabel='Compounds sorted by observed pKoff',title='Predicted ranks (1 = highest)')
            fig.colorbar(image,ax=bx,label='Predicted rank')
            fig.suptitle('MGCA final: '+group[0]['target_name']+' ('+panel+')')
            fig.tight_layout()
            dest=a.output_root/'figures'/panel;dest.mkdir(parents=True,exist_ok=True)
            for suffix in ['png','svg']:fig.savefig(dest/f'target_{index:02d}.{suffix}',dpi=220,bbox_inches='tight')
            plt.close(fig)

if __name__=='__main__':main()

import os
import pandas as pd
from cosinenet.evaluate import reranking
from cosinenet.data.dataset import QAdataset
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

met = ['P@1', 'MAP', 'MRR', 'Prec', 'Rec', 'F1', 'roc_auc', 'accuracy', 'answer triggering precision', 'answer triggering recall', 'answer triggering f1' ]
header = ['model', 'dataset', 'split'] + met
df = pd.DataFrame(columns=header)

for model in os.listdir(f"results/"):
    if model in ['compare', 'severyn', 'cosine']:
        for m2 in os.listdir(f"results/{model}"):
            m = f"{model}/{m2}"
            for dataset in os.listdir(f"results/{m}"):
                for split in os.listdir(f"results/{m}/{dataset}"):
                    print(m, dataset, split)
                    path = f"results/{m}/{dataset}/{split}"
                    dts = QAdataset(path,0,False)
                    metrics = reranking.evaluate(dts, th=0.5)
                    row = [f'{m}', dataset, split] + [metrics[m] for m in met]
                    df.loc[len(df)] = row
    else:
        for dataset in os.listdir(f"results/{model}"):
            for split in os.listdir(f"results/{model}/{dataset}"):
                print(model, dataset, split)
                path = f"results/{model}/{dataset}/{split}"
                dts = QAdataset(path,0,False)
                metrics = reranking.evaluate(dts, th=0.5)
                row = [f'{model}', dataset, split] + [metrics[m] for m in met]
                df.loc[len(df)] = row
print(df)
df.to_csv('result_summary.csv')
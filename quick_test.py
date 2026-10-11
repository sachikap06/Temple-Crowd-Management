import train_lstm, pandas as pd, numpy as np

df = pd.read_csv('output/crowd_dataset.csv')
Xtr,ytr,Xva,yva,Xte,yte,ti,nf = train_lstm.prepare_sequences(df, seq_length=10)
s1,s2,s3,s4,s5,sc = train_lstm.normalize_splits(Xtr,ytr,Xva,yva,Xte)
print(f'Scale: {sc}, n_feat: {nf}')
print(f'Train: {len(Xtr)}, Val: {len(Xva)}, Test: {len(Xte)}')

# Run a quick single experiment (30 epochs for speed)
vm,vr,m,w,ep,ps,tl,vl = train_lstm.run_experiment(s1,s2,s3,s4,s5,yte,ti,sc,'LSTM',64,2,0.001,'MSE',nf,max_ep=50,pat=15)
print(f'val_MAE={vm:.3f}')
print(f'test_MAE={m["mae"]:.4f}')
print(f'test_RMSE={m["rmse"]:.4f}')
print(f'R2={m["r2_score"]:.4f}')
print(f'Accuracy={m["accuracy"]*100:.2f}%')
print(f'F1={m["f1_score"]:.4f}')
print(f'epochs={ep}')

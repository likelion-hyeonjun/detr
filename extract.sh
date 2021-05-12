# LOG="train_imsitu_role190_adj.log"
LOG="swig_role6_e1d1_e_ffn.log"
 
cat $LOG | grep Aver > log
python -c "import pandas as pd; \
           log = pd.read_csv('log', sep=' ', header=None, index_col=0); \
           trn_log = log[::2].dropna(1); \
           dev_log = log[1::2].dropna(1); \
           pd.concat([trn_log.reset_index(), dev_log.reset_index()], axis=1).to_csv('loglog', sep=' ', index=False, header=False)"
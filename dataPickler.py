import pandas as pd
from sklearn.model_selection import train_test_split

real_data = pd.read_pickle('test_measure_hr.pkl')
syn_data = pd.read_pickle('CTGAN_synthetic_data1.pkl')




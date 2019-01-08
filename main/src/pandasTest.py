import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

# df = pd.DataFrame({'x':np.random.rand(10), 'y':np.random.rand(10)},
#                   index=list(string.ascii_lowercase[:10]))

df = pd.DataFrame({'x':np.asarray([1, 5.5, 7.5]), 'y':np.random.rand(3)}, index=('Event 1', 'Event 2', 'Event 3'))
print(df)
print(df.shape)

df2 = pd.read_csv('./main/bin/unique/hello_train_labels.csv')
print(df2)
print(df2.shape)
plt.figure()
for i in range(0, df.shape[0]):
    plt.axvline(df.x[i])
plt.figure()
for i in range(0, df2.shape[0]):
    plt.axvline(df2.Time[i], color=df2.Color[i], linestyle='-.')
    plt.annotate(df2.Event[i], xy=(df2.Time[i], 0), xytext=(5, 5), textcoords='offset points')
plt.show()


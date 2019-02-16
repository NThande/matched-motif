import matplotlib.pyplot as plt
import pandas as pd


def test_pandas():
    df2 = pd.read_csv('./main/bin/unique/hello_train_labels.csv')
    plt.figure()
    for i in range(0, df2.shape[0] - 1):
        plt.axvspan(df2.Time[i], df2.Time[i + 1], color=df2.Color[i], linestyle='-.')
        plt.annotate(df2.Event[i], xy=(df2.Time[i], 0), xytext=(5, 5), textcoords='offset points')
    plt.show()


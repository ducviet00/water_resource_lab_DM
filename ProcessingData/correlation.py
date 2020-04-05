import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

def plot_correlation(data):
    data_corr = data.corr()
    fig = plt.figure(figsize=(8,6))
    sb.heatmap(data_corr,vmax=1.0,center=0,fmt='0.2f',cbar=True,annot=True,
                cbar_kws={'shrink': .70,},square=True,linewidths=.5)
    fig.savefig('./Log/DataAnalysis/correlation.png')
    plt.show()

def plot_Q_H_data(data):
    fig = plt.figure(figsize=(10,6))
    
    fig.add_subplot(121)
    plt.plot(data.iloc[-2000:,5])
    plt.xlabel('Date')
    plt.ylabel('Q')
    plt.title('Discharge (Q)')

    fig.add_subplot(122)
    plt.plot(data.iloc[-2000:,-1])
    plt.xlabel('Date')
    plt.ylabel('H')
    plt.title('Water Level (H)')

    fig.savefig('./Log/DataAnalysis/real_data.png')
    plt.show()

def data_group():
    #data.index = pd.to_datetime(data['time'])
    data = pd.read_csv('../RawData/Kontum-daily.csv',header=0,index_col=0)
    data['time'] = pd.to_datetime(data['time'])
    g = data['time'].dt.to_period('M')
    group = data.groupby(g)

    monthly_avg = group.aggregate(np.mean)
    monthly_avg['time'] = monthly_avg.index

    #plt.plot(monthly_avg['time'],monthly_avg.iloc[:,5])
    # monthly_avg.plot('time','discharge')
    # plt.xlabel('Month')
    # plt.ylabel('Q')
    # plt.title('Discharge (Q)')
    # plt.savefig('./Log/DataAnalysis/data_avg_month_Q.png')
    # plt.show()
    # #plt.plot(monthly_avg['time'],monthly_avg.iloc[:,7])
    # monthly_avg.plot('time','water_level')
    # plt.xlabel('Month')
    # plt.ylabel('H')
    # plt.title('Water Level (H)')
    # plt.savefig('./Log/DataAnalysis/data_avg_month_H.png')
    # plt.show()

    return monthly_avg


if __name__ == '__main__':
    data = pd.read_csv('./RawData/Kontum-daily.csv',header=0,index_col=0)
    month = data_group()
    print(month.head())

    #data_analyse(data)
    #data = data.drop('time',axis=1)
    #data = data.to_numpy()
    #plot_Q_H_data(data)
    #print(data.shape)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:04:41 2018

@author: ishidukotaro
"""
from statistics import mean, median,variance,stdev
import pandas as pd
from sklearn.svm import SVC
import math
import itertools
from sklearn.mixture import GMM
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json,csv
import glob
import copy
from sklearn import datasets
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier

#大ガスからもらったCSVファイルを読み込んで余分な列を落として電力とタイムスタンプだけにする
#windowsizeごとにスライディングウィンドウしてベクトルのリストを返す
n=0
d_list=[0]*5
def readdata(date,homenumber,a,b):
    windowsize=10
    pvectorlist=[]
    filename="/Users/ishidukotaro/Desktop/study/program/homedata/"+date+'/data_'+homenumber+'_raw_2019'+date+'.csv'
    df = pd.read_csv(filename)
    df_power=df.loc[ : , ['timestamp','power']]
    df_power.dropna(inplace=True)
    df_power['timestamp']=pd.to_datetime(df_power['timestamp'],format='%Y/%m/%d  %H:%M:%S.000')
    df_power=df_power.reset_index(drop=True)
    dfvector=df_power['power'].values
    dfvector = dfvector.astype(np.int64)
    dflist=dfvector.tolist()
  
    dflist2=copy.deepcopy(dflist)
    dflist2.insert(0,0)
    #dflist2.insert(-1,0)
  

    i=0
    while i+windowsize<=len(df_power):
        plist=dflist[i:i+windowsize]
        p=np.r_[plist,[df_power['timestamp'][int(i+(windowsize/2))].hour]]
        pvectorlist.append(p)
        i+=1
    '''
    for i,item in enumerate(dflist2):
        if 0<i<len(dflist2)-1:
            dflist2[i]=(dflist2[i-1]+dflist2[i]+dflist2[i+1])/3
    '''

    plt.plot(dflist2[a:b])
    return dflist2[a:b]

def read_smasockdata(date,nextdate,id):
    filename="/Users/ishidukotaro/Desktop/study/program/y-homedata/"+date+'smasock-'+str(id)+'.csv'
    d = pd.read_csv(filename)
    df=d.iloc[360: , 1]
    filename2="/Users/ishidukotaro/Desktop/study/program/y-homedata/"+nextdate+'smasock-1.csv'
    d2 = pd.read_csv(filename2)
    df2=d2.iloc[ :360 , 1]

    df=df=pd.concat([df, df2])
    df.reset_index(drop=True, inplace=True)

    return df
      
     
#スマソケでとったデータを読み込んで各家電の電力の総和のリストを返す
def make_smasockdatalist(date,nextdate,a,b):
    colorcode=["","b","g","r",'#ffff33',"c","m",'#ff7f00','#f781bf',"y","k", '#377eb8']
    df=read_smasockdata(date,nextdate,1)
    dfpowervalue=df.values
    dfpowerlist=dfpowervalue.tolist()
    plt.plot(dfpowerlist[a:b])
    i=2
    while i<12:
        if i!=7 and i!=8 and i!=11:
            d_power=read_smasockdata(date,nextdate,i)  
            df=pd.concat([df, d_power], axis=1)
            dfpowervalue=d_power.values
            dfpowerlist=dfpowervalue.tolist()
            plt.plot(dfpowerlist[a:b],colorcode[i])
        i+=1
    
    #各行の和を算出
    dfsum=df.sum(axis=1)
    df=pd.concat([df, dfsum], axis=1)
    dfsumvalue=dfsum.values
    dflist=dfsumvalue.tolist()
    dflist=dflist[a:b]

    return dflist

#総電力のベクトルを受け取り家電を推定
def estimate(list,date):
    '''
    for item in list:
        print(item)
        
    '''
    featurelist=[]
    peaklist=[]
    peakindexlist=[]
    j=0
    i=0
    N=len(list)
    dt = 30
    t = np.arange(0, N*dt, dt)
    grouplist=[]
    valuelist=[]#長方形の高さの値を格納
    g=0
    #差をとる
    list_2=[]
    while i<len(list)-1:
        list_2.insert(i,list[i+1]-list[i])
        i+=1      
    list_2.append(list[-1])
    #移動平均
    list_3=[]
    list_3.append((list[0]+list[1])/2)
    i=1
    while i<len(list)-1:
        list_3.insert(i,(list[i-1]+list[i]+list[i+1])/3)
        i+=1      
    list_3.append((list[-1]+list[-2])/2)
    #plt.plot(list)
    #plt.plot(list_3)
    
    i=1
    peaklist.append(0)
    while i<len(list)-1:
        if list[i-1]<list[i]-100 :
            peaklist.insert(i,1)
        elif list[i-1]-100>list[i]:
            peaklist.insert(i,-1)
        else:
            peaklist.insert(i,0)
        i+=1
    peaklist.append(0)
                
    passlist=make_square(list,valuelist,peakindexlist,date)
        
        
    return passlist


def make_square(list,valuelist,peakindexlist,date):
    square_list=[]
    variation_list=[]
    variation_list_list=[]
    highpasslist = [0] * len(list)
    lowpasslist = [0] * len(list)
    
    variation_list.append([10000,0,0])
    i=1
    while i<len(list)-1:
        if i+1<len(list) and list[i-1]<100 and list[i]<100 and list[i+1]<100 and variation_list[-1][0]!=10000:
            variation_list.append([10000,i,i])
        power=list[i]-list[i-1] 
        if power>=100:#大きい変化を見つける
            j=i
            k=i
            while list[j-1]-list[j-2]>=1:
                power=power+list[j-1]-list[j-2]
                j-=1
            while list[k+1]-list[k]>1:
                power=power+list[k+1]-list[k]
                k+=1
                if k>=len(list)-1:
                    break
            variation_list.append([power,j,k])
        elif -100>=power:
            j=i
            k=i
            while list[j-1]-list[j-2]<=-1:
                power=power+list[j-1]-list[j-2]
                j-=1
            while list[k+1]-list[k]<-1:
                power=power+list[k+1]-list[k]
                k+=1
                if k>=len(list)-1:
                    break
            variation_list.append([power,j,k])
        else:   
           None
        i+=1
        
    for item in variation_list:
        for item2 in variation_list:
            if item[0]!=10000 and item2[0]!=10000:
                if item[1]==item2[1] and item[2]!=item2[2] and item2[0]!=0:
                    if (item[0]>0 and item[0]>item2[0])or(item[0]<0 and item[0]<item2[0]):
                        item2[0]=0
                    else:
                        item[0]=0
                elif item[2]==item2[2] and item[1]!=item2[1] and item2[0]!=0:
                    if (item[0]>0 and item[0]>item2[0])or(item[0]<0 and item[0]<item2[0]):
                        item2[0]=0
                    else:
                        item[0]=0
                elif item[2]==item2[2] and item[1]==item[1]and item[0]!=item2[0]:
                    if (item[0]>0 and item[0]>item2[0])or(item[0]<0 and item[0]<item2[0]):
                        item2[0]=0
                    else:
                        item[0]=0
    i=-1    
    variation_list = get_unique_list(variation_list)
    for item in variation_list:
        if item[0]==10000:
            i+=1
            variation_list_list.append([])
        else:
            variation_list_list[i].append(item)
    '''
    for vlist in variation_list_list:
        for item in vlist:
            if item[0]!=0:
                print(item)
    '''
    
    variation_list_list_list=[]
    square_list_list=[]
    variation_list_list_list.append(variation_list_list)
    square_list_list.append(square_list)
    
    for i in range(0,8):
        variation_list_list_list.append(copy.deepcopy(variation_list_list))
        square_list_list.append(copy.deepcopy(square_list))
    
    for i in reversed(range(0,8)):
        make_square_step1(variation_list_list_list[i],square_list_list[i],i)
        #passlist=high_pass(list,square_list_list[i],i)
        square_print(list,square_list_list[i],i,date)

   
    '''
    for i,item in enumerate(passlist):
        if 0<i<len(passlist)-1:
            passlist[i]=(passlist[i-1]+passlist[i]+passlist[i+1])/3
    '''
    #square(filter(passlist))
    
    #return passlist
      

def high_pass(list,square_list,mode):
    highpass_list=[]
    plt.figure()
    #plt.figure()
    #ax1 = fig.add_subplot(111)
    for i,item in enumerate(square_list):
        if item[3]-item[2]<100 and item[0]>100:
            highpass_list.append(item)
            
    peaklist=[0] * len(list)
    bandpass_list = [0] * len(list)
    for item in highpass_list:
        printlist = [0] * len(list)
        if item[0]!=0 and item[1]!=0:
            v=np.linspace(item[0],item[1],item[3]-item[2])
            for i in range(item[2], item[3]):
                bandpass_list[i] = bandpass_list[i]+v[i-item[2]]
                peaklist[i]=1
                if printlist[i]+v[i-item[2]]>500:
                    printlist[i] = printlist[i]+v[i-item[2]]
        y2 = printlist
        ln2=plt.plot(y2)
    plt.show()
    passlist=(np.array(list)-np.array(bandpass_list)).tolist()
    passlist2=copy.deepcopy(passlist)
    
    for i,item in enumerate(passlist):
        if peaklist[i]==1:
            j=i
            while peaklist[j]==1:
                j-=1
            passlist2[i]=passlist2[j]
            
    #plt.plot(passlist2)
    return passlist2
        
        
def make_square_step1(variation_list_list,square_list,mode):
    for vlist in variation_list_list:
        for i,item in enumerate(vlist):
            if item[0]>0:
                if i!=0 and vlist[i-1][0]<0 and (item[1]-vlist[i-1][1])<30 and vlist[i-1][0]+item[0]<(item[0]/3):
                    #square_list.append([vlist[i-1][0],-item[0],vlist[i-1][1],item[2]])
                    item[0]=0
                    vlist[i-1][0]=0
            elif item[0]<0:
                
                templist=[]
                k=0
                while k<vlist.index(item)-1:
                    if vlist[k][0]!=0:
                        templist.append([vlist[k][0],k])
                    k+=1
                if len(templist)>0:
                    k=templist[0][1]

                if len(templist)==1 and  (item[1]-vlist[k][1])<5 and templist[0][0]>0 and (templist[0][0]+item[0])>0:
                    square_list.append([(vlist[k][0]+item[0]),-item[0],vlist[k][1],item[2]])
                    vlist[k][0]+=item[0]
                    item[0]=0
                
                if mode<4:
                    forlist=range(0,vlist.index(item))
                else:
                    forlist=reversed(range(0,vlist.index(item)))
                
                for j in forlist:
                    if vlist[j][0]!=0 and 0<(item[0]+vlist[j][0])<(-item[0]/3):
                        square_list.append([vlist[j][0],-item[0],vlist[j][1],item[2]])
                        item[0]=0
                        vlist[j][0]=0
            else:
                None
    
    lists=make_square_step2(variation_list_list,square_list,mode)
    variation_list_list=lists[0]
    square_list=lists[1]
    return variation_list_list,square_list

#二つ同時に上がるor下がる
def make_square_step2(variation_list_list,square_list,mode):   
    mode=mode%4
    
    for vlist in variation_list_list:
        for item in vlist:
            
            n=vlist.index(item)-1
            m=n+2
            if n<0:
                n=1
            if m>len(vlist):
                m=len(vlist)
            if mode==0:
                list1=range(0,vlist.index(item))
                list2=range(vlist.index(item)+1,len(vlist))
            elif mode==1:
                list1=list(reversed(range(0,vlist.index(item))))
                list2=range(vlist.index(item)+1,len(vlist))
            elif mode==2:
                list1=range(0,vlist.index(item))
                list2=list(reversed(range(vlist.index(item)+1,len(vlist))))
            else:
                list1=list(reversed(range(0,vlist.index(item))))
                list2=list(reversed(range(vlist.index(item)+1,len(vlist))))
            if item[0]<0:
                for i in list1:
                    for j in list1:
                        if i!=j and vlist[i][0]>0 and vlist[j][0]>0 and abs(vlist[i][0]+vlist[j][0]+item[0])<(-item[0]/3):
                            square_list.append([vlist[i][0],vlist[i][0],vlist[i][1],item[2]])
                            square_list.append([vlist[j][0],vlist[j][0],vlist[j][1],item[2]])
                            vlist[i][0]=0
                            vlist[j][0]=0
                            item[0]=0
            #まとめて上がって複数回で下がった場合
            if item[0]>0:
                for i in list2:
                    for j in list2:
                        if i!=j and vlist[i][0]<0 and vlist[j][0]<0 and abs(vlist[i][0]+vlist[j][0]+item[0])<(item[0]/3):
                            square_list.append([-vlist[i][0],-vlist[i][0],item[1],vlist[i][2]])
                            square_list.append([-vlist[j][0],-vlist[j][0],item[1],vlist[j][2]])
                            vlist[i][0]=0
                            vlist[j][0]=0
                            item[0]=0
                            
    
    lists=make_square_step3(variation_list_list,square_list)
    variation_list_list=lists[0]
    square_list=lists[1]
    
    
    return variation_list_list,square_list

def make_square_step3(variation_list_list,square_list):
    
    for vlist in variation_list_list:
        for item in vlist:
            #複数回で上がってまとめて下がった時
            for i in range(0,vlist.index(item)-1):
                for j in range(0,vlist.index(item)-1):
                    for k in range(0,vlist.index(item)-1):
                        if i!=j and j!=k and i!=k and vlist[i][0]>0 and vlist[j][0]>0 and vlist[k][0]>0 and abs(vlist[i][0]+vlist[j][0]+vlist[k][0]+item[0])<(-item[0]/3):
                            square_list.append([vlist[i][0],vlist[i][0],vlist[i][1],item[2]])
                            square_list.append([vlist[j][0],vlist[j][0],vlist[j][1],item[2]])
                            square_list.append([vlist[k][0],vlist[k][0],vlist[k][1],item[2]])
                            vlist[k][0]=0
                            vlist[i][0]=0
                            vlist[j][0]=0
                            item[0]=0
            #まとめて上がって複数回で下がった場合
            for i in range(vlist.index(item)+1,len(vlist)):
                for j in range(vlist.index(item)+1,len(vlist)):
                    for k in range(vlist.index(item)+1,len(vlist)):
                        if i!=j and j!=k and i!=k and vlist[i][0]<0 and vlist[j][0]<0 and vlist[k][0]<0 and abs(vlist[i][0]+vlist[j][0]+vlist[k][0]+item[0])<(item[0]/3):
                            square_list.append([-vlist[i][0],-vlist[i][0],item[1],vlist[i][2]])
                            square_list.append([-vlist[j][0],-vlist[j][0],item[1],vlist[j][2]])
                            square_list.append([-vlist[k][0],-vlist[k][0],item[1],vlist[k][2]])
                            vlist[k][0]=0
                            vlist[i][0]=0
                            vlist[j][0]=0
                            item[0]=0
                            
    

    
    for vlist in variation_list_list:
        list2 = [x for x in vlist if x[0] != 0]
        if len(list2)==2 and list2[0][0]>0 and list2[1][0]<0 and abs(list2[0][0]+list2[1][0])<300:
            square_list.append([list2[0][0],-list2[1][0],list2[0][1],list2[1][2]])
            
    return variation_list_list,square_list
    
def square_print(list,square_list,id,date):
    colorcode=["","b","g","r", '#377eb8',"c","m","","","y","k"]
    plt.figure()
    #print(list,id)
    #fig2=plt.figure()
    #ax2 = fig2.add_subplot(111)
    y1 = list
    #ln1=ax1.plot(y1,'C0',label=r'power')   
    #ax1.set_xlabel('t')
    #ax1.set_ylabel(r'power')
    #ax1.grid(True)
    
    knn_appliance(square_list,date)
    
    bandpass_list = [0] * len(list)
    for item in square_list:
        printlist = [0] * len(list)
        if item[0]!=0 and item[1]!=0:
            v=np.linspace(item[0],item[1],item[3]-item[2])
            for i in range(item[2], item[3]):
                bandpass_list[i] = bandpass_list[i]+v[i-item[2]]
                printlist[i] = printlist[i]+v[i-item[2]]
        y2 = printlist
        ln2=plt.plot(y2,colorcode[item[-1]])
    plt.show()
    simiraly=0
    #類似度計算
    for i in range(0,len(list)):
        j=abs(list[i]-bandpass_list[i])
        simiraly=simiraly+j
    
    print(id,simiraly)
    knn_appliance(square_list,date)
    return bandpass_list


#fftしてフィルタする
def filter(list):
    plt.figure()
    N=len(list)
    dt = 30#サンプリング周期
    t = np.arange(0, N*dt, dt) #時間軸
    #listをfft
    F = np.fft.fft(list)
    F_abs = np.abs(F)
    F_abs_amp = F_abs/N*2 # 交流成分はデータ数で割って2倍する
    F_abs_amp[0]=F_abs_amp[0]/2 # 直流成分（今回は扱わないけど）は2倍不要
    fq = np.linspace(0, 1.0/dt, N)
    #plt.xlabel('freqency(Hz)', fontsize=14)
    #plt.ylabel('signal amplitude', fontsize=14)
    #plt.plot(fq[:int(N/2)+1], F_abs_amp[:int(N/2)+1]) 
    #plt.hlines(y=[0.2],xmin=0, xmax=0.035, colors='r', linestyles='dashed')
    #plt.plot(fq, F_abs_amp)
    
    
    #フィルタ(周波数)
    F2 = np.copy(F) # FFT結果コピー
    fc1 =0.00025 # カットオフ（周波数）
    fc2 =0.0001
    
    F2[(fq > fc1)] = 0 # カットオフを超える周波数のデータをゼロにする（ノイズ除去）
    #F2[(fq < fc2)] = 0
    
    F2_abs = np.abs(F2)
    # 振幅をもとの信号に揃える
    F2_abs_amp = F2_abs / N * 2 # 交流成分はデータ数で割って2倍
    F2_abs_amp[0] = F2_abs_amp[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要
    
    F2_ifft = np.fft.ifft(F2)
    F2_ifft_real = F2_ifft.real * 2
    plt.plot(t, list, label='original')
    plt.plot(t, F2_ifft_real, c="r", linewidth=1, alpha=0.7, label='filtered')
    plt.legend(loc='best')
    plt.xlabel('time(sec)', fontsize=14)
    plt.ylabel('singnal', fontsize=14)
    '''
    #フィルタ(振幅)
    F3 = np.copy(F) # FFT結果コピー
    F3 = np.copy(F) # FFT結果コピー
    ac = 80 # 振幅強度の閾値
    F3[(F_abs_amp < ac)] = 0 # 振幅が閾値以上はゼロにする（ノイズ除去）
    F3_ifft = np.fft.ifft(F3) # IFFT
    F3_ifft_real = F3_ifft.real # 実数部の取得
    # グラフ（オリジナルとフィルタリングを比較）
    #plt.plot(t, list, label='original')
    plt.plot(t,F3_ifft_real, c="green", linewidth=1, alpha=0.7, label='filtered')
    plt.legend(loc='best')
    plt.xlabel('time(sec)', fontsize=14)
    plt.ylabel('singnal', fontsize=14)
    '''
    return F2_ifft_real

def identify_appliance(square_list):
    appliances_list=["電子レンジorオーブン","テレビ","電気ポット","洗濯機","ドライヤー","ストーブ","加湿器"]
    for square in square_list:
        power=(square[0]+square[1])/2
        time=square[3]-square[2]
        hour=round((square[2]+square[3])/2*20/3600)
        print(square,power,time,hour)

    
def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def make_teacherdata():
    datelist=["2018-12-26-","2018-12-27-","2018-12-28-","2018-12-29-","2018-12-30-","2018-12-31-","2019-01-01-",
              "2019-01-03-","2019-01-04-","2019-01-05-","2019-01-06-","2019-01-07-","2019-01-08-","2019-01-09-",
              "2019-01-10-","2019-01-11-","2019-01-12-","2019-01-13-","2019-01-14-","2019-01-15-","2019-01-16-",
              "2019-01-17-","2019-01-18-","2019-01-19-","2019-01-20-"]
   
    
    teacherdata_list=[]
    lowteacherdata_list=[]
    highteacherdata_list=[]
    for i in range(0,len(datelist)-1):
        for j in range(1,12):
            if j!=7 and j!=8 and j!=11:
                for tdata in (read_teacherdata(datelist[i],datelist[i+1],j))[0]:
                    teacherdata_list.append(tdata)
                for tdata in (read_teacherdata(datelist[i],datelist[i+1],j))[1]:
                    lowteacherdata_list.append(tdata)
                for tdata in (read_teacherdata(datelist[i],datelist[i+1],j))[2]:
                    highteacherdata_list.append(tdata)    
                
    return teacherdata_list,lowteacherdata_list,highteacherdata_list
        
def read_teacherdata(date,nextdate,id):
    teacherdata_list=[]
    lowteacherdata_list=[]
    highteacherdata_list=[]
    df=read_smasockdata(date,nextdate,id)
    dfpowervalue=df.values
    dfpowerlist=dfpowervalue.tolist()
    flag=0
    i=1
    while i<len(dfpowerlist)-1:
        power=dfpowerlist[i]-dfpowerlist[i-1]
        if flag==0 and power>=100:#大きい変化を見つける
            j=i
            k=i
            while dfpowerlist[j-1]-dfpowerlist[j-2]>=1:
                power=power+dfpowerlist[j-1]-dfpowerlist[j-2]
                j-=1
            while dfpowerlist[k+1]-dfpowerlist[k]>1:
                power=power+dfpowerlist[k+1]-dfpowerlist[k]
                k+=1
                if k>=len(dfpowerlist)-1:
                    break
            power1=power
            flag=1
        elif flag==1 and -(power1)/1.5>=power:
            l=i
            m=i
            while dfpowerlist[l-1]-dfpowerlist[l-2]<=0:
                power=power+dfpowerlist[l-1]-dfpowerlist[l-2]
                l-=1
            while dfpowerlist[m+1]-dfpowerlist[m]<-1:
                power=power+dfpowerlist[m+1]-dfpowerlist[m]
                m+=1
                if m>=len(dfpowerlist)-1:
                    break
            teacherdata_list.append([power1,-power,j+2,m+2,id,date])
            if id==1 or id==3 or id==5 or id==10:
                lowteacherdata_list.append([power1,-power,j+2,m+2,1,date])
                highteacherdata_list.append([power1,-power,j+2,m+2,id,date])
            else:
                lowteacherdata_list.append([power1,-power,j+2,m+2,id,date])

            flag=0
            i=m
        else:   
           None
        i+=1
        
    return teacherdata_list,lowteacherdata_list,highteacherdata_list
    
def knn_appliance(square_list,date):
    #print(square_list)
    teacherdata_lists=make_teacherdata()
    teacherdata_list=teacherdata_lists[0]
    lowteacherdata_list=teacherdata_lists[1]
    highteacherdata_list=teacherdata_lists[2]
    #print("teacherdata_list",teacherdata_list)
    traindata_list=[]
    trainid_list=[]
    testdata_list=[]
    maxtime=0
    maxpower=0
    maxdifferenceheight=0
    maxgradient=0
    #洗濯機判別
    
    #矩形波統合
    
    #訓練データ読み込み
    for teacherdata in teacherdata_list:
        if teacherdata[5]!=date:
            maxpower=max(maxpower,(teacherdata[0]+teacherdata[1])/2)
            maxtime=max(maxtime,teacherdata[3]-teacherdata[2])
            if teacherdata[0]-teacherdata[1]>500:
                print(teacherdata[0],teacherdata[1],teacherdata)
            maxdifferenceheight=max(maxdifferenceheight,teacherdata[0]-teacherdata[1])
            hour=round((teacherdata[2]+teacherdata[3])/2*20/3600)
            
    print(maxpower,maxtime,maxdifferenceheight)
    for teacherdata in teacherdata_list:
        if teacherdata[5]!=date:
            power=((teacherdata[0]+teacherdata[1])/2)/maxpower
            time=((teacherdata[3]-teacherdata[2])/maxtime)
            differenceheight=((teacherdata[0]-teacherdata[1])/maxdifferenceheight)
            hour=round((teacherdata[2]+teacherdata[3])/2*20/3600)
            hoursin=math.sin(math.radians(hour*15))/100
            hourcos=math.cos(math.radians(hour*15))/100
            traindata=[power,time,differenceheight,hoursin,hourcos]
            traindata_list.append(list(map(lambda x: x * 100, traindata)))
            trainid_list.append(teacherdata[4])
    
    for square in square_list:
        power=((square[0]+square[1])/2)/maxpower
        time=((square[3]-square[2])/maxtime)
        differenceheight=((square[0]-square[1])/maxdifferenceheight)
        hour=round((square[2]+square[3])/2*20/3600)
        hoursin=math.sin(math.radians(hour*15))/100
        hourcos=math.cos(math.radians(hour*15))/100
        testdata=[power,time,differenceheight,hoursin,hourcos]
        testdata_list.append(list(map(lambda x: x * 100, testdata)))
        
        
    #print(trainid_list)
    knc = KNeighborsClassifier(n_neighbors=5)
    knc.fit(traindata_list, trainid_list)
    
    #print(testdata_list)
    '''
    for i in range(0,len(testdata_list)):
        print(testdata_list[i],knc.kneighbors(testdata_list,5)[i])
    '''
    Y_pred = knc.predict(testdata_list)
    #print("before",square_list)
    for i,pred in enumerate(Y_pred):
        square_list[i].append(pred)
    #print("after",square_list)
    
    
def read_eventteacherdata(date,nextdate,id):
    #plt.figure()
    teacherdata_list=[]#イベントリストのリスト
    teacherlabel_list=[]
    df=read_smasockdata(date,nextdate,id)
    dfpowervalue=df.values
    dfpowerlist=dfpowervalue.tolist()
    flag=0
    i=1
    event_list=[]
    start=0
    value=0
    start_list=[]
    
    while i<len(dfpowerlist)-1:
        power=dfpowerlist[i]-dfpowerlist[i-1]
        if power>=50:#大きい変化を見つける
            if flag==0:
                start=i
                start_list.append(start)
                flag=1
            if i!=0:
                event_list.append([value,i-start-1])
                value=0
            event_list.append([power,i-start])
            
        elif -50>=power and flag==1:
            if i!=0 and flag!=0:
                event_list.append([value,i-start-1])
                value=0
            event_list.append([power,i-start])
            if dfpowerlist[i]<100:
                teacherdata_list.append(event_list)
                teacherlabel_list.append(id)
                event_list=[]
                flag=0
        else:
            value=value+power
        i+=1
        
    teachervaluedata_list=[]   
    for teacherdata in teacherdata_list:
        valuedata_list=[0]
        value=0
        #if id==3:
            #print(range(0,teacherdata[-1][1]+1))
        for j in range(0,teacherdata[-1][1]+1):
            for event in teacherdata:
                if event[1]==j:
                    value=value+event[0]    
            valuedata_list.append(value)
        valuedata_list.pop(0)
        #if id==3:
            #print(teacherdata,valuedata_list,range(0,teacherdata[-1][1]+1))
        amplitudemax=max(valuedata_list)
        amplitudeavg=sum(valuedata_list)/len(valuedata_list)
        opetime=len(valuedata_list)
        wave=(np.array(valuedata_list)/amplitudemax).tolist()
        hour=round(start_list.pop(0)*20/3600)
        hoursin=math.sin(math.radians(hour*15))
        hourcos=math.cos(math.radians(hour*15))
        teachervaluedata_list.append([wave,[amplitudemax,amplitudeavg],[hoursin,hourcos],opetime])
        #if id==1:
            #plt.plot(valuedata_list)
    #if id==3:
        #print(date,teacherdata_list,teachervaluedata_list)
    
    return teachervaluedata_list,teacherlabel_list
def make_eventteacherdata():
    datelist=["2018-12-26-","2018-12-27-","2018-12-28-","2018-12-29-","2018-12-30-","2018-12-31-","2019-01-01-",
              "2019-01-03-","2019-01-04-","2019-01-05-","2019-01-06-","2019-01-07-","2019-01-08-","2019-01-09-",
              "2019-01-10-","2019-01-11-","2019-01-12-","2019-01-13-","2019-01-14-","2019-01-15-","2019-01-16-",
              "2019-01-17-","2019-01-18-","2019-01-19-","2019-01-20-"]
   
    
    teacherdata_list=[]
    for i in range(0,len(datelist)-1):
        for j in range(1,12):
            if j!=4 and j!=7 and j!=8 and j!=11:
                teacherdata=read_eventteacherdata(datelist[i],datelist[i+1],j)
                forlist=teacherdata[0]
                for k,tdata in enumerate(forlist):
                    teacherdata_list.append([tdata,j])
    
    return teacherdata_list

def make_eventtree(list,maxindex,teacherdatalist):
    plt.figure()
    plt.plot(list)
    #variation_list=[[0,0],[300,1],[400,2],[-200,3],[150,4]]
    global d_list
    variation_list=[]
    for i,value in enumerate(list):
        if i!=0:
            variation_list.append([list[i]-list[i-1],i])
        if i!=0 and(100<list[i]-list[i-1] or list[i]-list[i-1]<-100):
            print(list[i]-list[i-1],i)      
    global n
    rootnode=Treenode(None,0,{},-1,0,0,n)
    n+=1
    list_list=[]
    l=[]
    '''
    for i in range(2,len(list)):
        if list[i-2]<5 and list[i-1]<5 and list[i]<5:
            list_list.append(l)
            l=[]
        else:
            l.append(variation_list[i])
    
    list_list2=[]
    for li in list_list:
        if len(li)!=0:
            list_list2.append(li)
    '''     
    
    value=0
    for variation in variation_list:
        if variation[0]>50 or variation[0]<-50:
            d_list=[0]*20
            flag=search_leaf(rootnode,value,variation[1]-1,teacherdatalist)
            delete_duplication(rootnode,[])
            check_dtw(rootnode)
            if flag:
                prun_node(rootnode)
            print("printleafnode")
            print_leafnode(rootnode)
            d_list=[0]*20
            flag=search_leaf(rootnode,variation[0],variation[1],teacherdatalist)
            delete_duplication(rootnode,[])
            check_dtw(rootnode)
            if flag:
                prun_node(rootnode)
            print("printleafnode")
            print_leafnode(rootnode)
            value=0
        else:
            value=value+variation[0]
    print_tree(rootnode,[],[],maxindex)
    
def add_event(node,value,time,teacherdatalist):
    #print("value",value,"time",time)
    global n
    if value==0:
        return False
    if value>0:
        for id in node.appliance_dict.keys():
            if node.appliance_dict[id][3]!=1:
                if value<1000:
                    appliance_dict=copy.deepcopy(node.appliance_dict)
                    appliance_dict[id][0]=appliance_dict[id][0]+value
                    eventnode=Treenode(node,node.id,appliance_dict,id,value,time,n)
                    n+=1
                    check_validity([id,value,time],eventnode,[],teacherdatalist,eventnode)
                    calclate_likelihood(eventnode)
                    node.child_list.append(eventnode)
        if value>50:
            appliance_dict=copy.deepcopy(node.appliance_dict)
            sorted(appliance_dict.items(), key=lambda x: x[0])
            id=node.id
            appliance_dict[id]=[value,0,0,0]
            eventnode=Treenode(node,id+1,appliance_dict,id,value,time,n)
            n+=1
            check_validity([id,value,time],eventnode,[],teacherdatalist,eventnode)
            calclate_likelihood(eventnode)
            node.child_list.append(eventnode)
    elif value!=0:
        #一つが落ちた
        for id in node.appliance_dict.keys():
            if node.appliance_dict[id][3]!=1:
                appliance_dict=copy.deepcopy(node.appliance_dict)
                appliance_dict[id][0]=appliance_dict[id][0]+value
                eventnode=Treenode(node,node.id,appliance_dict,id,value,time,n)
                if appliance_dict[id][0]<-100:
                    eventnode.delete=1
                n+=1
                check_validity([id,value,time],eventnode,[],teacherdatalist,eventnode)
                calclate_likelihood(eventnode)
                node.child_list.append(eventnode)
                if eventnode.appliance_dict[id][0]<50:
                    eventnode.appliance_dict[id][3]=1
                        
                        
        #二つ同時に落ちた時
        if value<-100:
            for id in node.appliance_dict.keys():
                for id2 in node.appliance_dict.keys():
                    if node.appliance_dict[id][3]!=1 and node.appliance_dict[id2][3]!=1 and id!=id2:
                        appliance_dict=copy.deepcopy(node.appliance_dict)
                        variation1=-appliance_dict[id][0]
                        appliance_dict[id][0]=appliance_dict[id][0]+variation1
                        eventnode=Treenode(node,node.id,appliance_dict,id,variation1,time,n)
                        check_validity([id,value,time],eventnode,[],teacherdatalist,eventnode)
                        n+=1
                        appliance_dict2=copy.deepcopy(eventnode.appliance_dict)
                        appliance_dict2[id2][0]=appliance_dict2[id2][0]+(value-variation1)
                        eventnode2=Treenode(eventnode,eventnode.id,appliance_dict2,id2,value-variation1,time,n)
                        check_validity([id2,value-variation1,time],eventnode2,[],teacherdatalist,eventnode2)
                        n+=1
                        if eventnode.appliance_dict[id][0]<100:
                            eventnode.appliance_dict[id][3]=1
                            eventnode2.appliance_dict[id][3]=1
                        if eventnode2.appliance_dict[id2][0]<100:
                            eventnode2.appliance_dict[id][3]=1
                        if appliance_dict[id][0]<-100 or variation1<value or appliance_dict2[id2][0]<-100:
                            eventnode.delete=1
                            eventnode2.delete=1
                        calclate_likelihood(eventnode2)
                        node.child_list.append(eventnode)
                        eventnode.child_list.append(eventnode2)
    return True

def print_tree(node,event_list,node_list,maxindex):
    if len(node.child_list)==0:
        plot_event(make_event_list(node,[],[]),maxindex,node)
        return 0
    for child in node.child_list:
        print_tree(child,event_list,node_list,maxindex)
    return 0
    
def make_event_list(node,event_list,node_list):
    event_list.insert(0,[node.appliance,node.value,node.time])
    node_list.insert(0,[node.appliance_dict,node.appliance,node.value,node.time])
    if node.parent!=None:
        make_event_list(node.parent,event_list,node_list)
        return event_list
    #print(node_list)

    return event_list
    
def search_leaf(node,value,time,teacherdatalist):
    if value==0:
        return False
    if node.child_list==[]:
        add_event(node,value,time,teacherdatalist)
        return True
    for child in node.child_list:
        search_leaf(child,value,time,teacherdatalist)
    return True
    
def plot_event(event_list,maxindex,node):
    colorcode=["","b","g","r",'#ffff33',"c","m",'#ff7f00','#f781bf',"y","k", '#377eb8']
    appliances_list=["","電子レンジ","テレビ","電気ポット","洗濯機","ドライヤー","ストーブ","ガスストーブ","掃除機","加湿器","オーブン",""]
    plt.figure()
    newevent_list=copy.deepcopy(event_list)
    t=range(0,maxindex)
    i=0
    newevent_list.pop(0)
    nd=node
    while(nd.parent!=None):
        print(nd.appliance,nd.value,nd.time)
        nd=nd.parent
    while len(newevent_list)>0:
        plotevent_list=[]
        for event in event_list:
            if int(event[0])==i:
                plotevent_list.append(event)
                newevent_list.remove(event)
        valuelist=[0] * (maxindex)
        value=0
        if len(plotevent_list)!=0:
            for j in t:
                for event in plotevent_list:
                    if event[2]==j:
                        value=value+event[1]     
                    valuelist[j]=value
        print(node.appliance_dict,[node.appliance_dict[i][2]])
        plt.plot(t,valuelist,colorcode[node.appliance_dict[i][2]])
        i+=1
    d=0

    print(node.number)
    for id in node.appliance_dict.keys():
        appliance=appliances_list[node.appliance_dict[id][2]]
        print(node.appliance_dict[id],appliance)
        d=d+node.appliance_dict[id][1]
    print(d)
    
def check_validity(event,node,event_list,teacherdatalist,leafnode):
    if node.parent==None:
        event_list.insert(0,[node.appliance,node.value,node.time])
        return check_event(event,event_list,teacherdatalist,leafnode)
    event_list.insert(0,[node.appliance,node.value,node.time])
    validity=check_validity(event,node.parent,event_list,teacherdatalist,leafnode)
    return validity
    
def check_event(event,event_list,teacherdatalist,node):
    global d_list
    newevent_list=copy.deepcopy(event_list)
    difference=0
    valuelist=[]
    value=0
    newevent_list.pop(0)
    noweventid=event[0]
    nowevent_list=[]         
    id=-1
    for e in event_list:
        if int(e[0])==noweventid:
            nowevent_list.append(e)
        '''
        for j in range(0,len(checkevent_list)-1):
            lastevent=nowevent_list[-1]          
            if lastevent[1]>800 and event[2]-lastevent[2]>30:#800Wで10分続く
                return False,0,0
            if event[2]-lastevent[2]>900:#５時間続く
                print(noweventid,checkevent_list,event)
                #print("2",checkevent_list,event[2],lastevent[2]>30)
                return False,0,0
        '''
     #相違度計算
    
    start=nowevent_list[0][2]
    for j in range(0,nowevent_list[-1][2]+1-start):
        for event in nowevent_list:
            if event[2]-start==j:
                value=value+event[1]     
        valuelist.append(value)
    
    amplitudemax=max(valuelist)
    amplitudeavg=sum(valuelist)/len(valuelist)
    opetime=len(valuelist)
    wave=np.array(valuelist)/amplitudemax
    hour=round(event[2]*20/3600)
    hoursin=math.sin(math.radians(hour*15))
    hourcos=math.cos(math.radians(hour*15))
    
    f=0
    for teacherdata in teacherdatalist:
        tdata=copy.deepcopy(teacherdata)
        if valuelist[-1]<50:
            simdtw=-calc_dtw(wave,tdata[0][0][0:len(valuelist)])[-1][-1][0]
            dtw=calc_dtw(wave,tdata[0][0])[-1][-1][0]
            simA=cos_sim([amplitudemax,amplitudeavg],tdata[0][1])
            simtp=cos_sim([hoursin,hourcos],tdata[0][2])
            simti=min(opetime,tdata[0][3])/max(opetime,tdata[0][3])
            #sim=dtw
            sim=simdtw+simA+simtp+simti
            if f==0:
                maxsim=sim
                id=tdata[1]
                f=1
            if f==1 and sim>maxsim:
                maxsim=sim
                id=tdata[1]
        elif len(tdata[0][0])-len(valuelist)>-100:             
            #if len(valuelist)-len(tdata[0][0])>0:
                #while len(valuelist)-len(tdata[0][0])!=0:
                    #tdata[0][0].append(0)
            simdtw=-calc_dtw(wave,tdata[0][0][0:len(valuelist)])[-1][-1][0]
            dtw=calc_dtw(wave,tdata[0][0][0:len(valuelist)])[-1][-1][0]
            simA=cos_sim([amplitudemax,amplitudeavg],tdata[0][1])
            simtp=cos_sim([hoursin,hourcos],tdata[0][2])
            simti=1
            #sim=dtw
            sim=simdtw+simA+simtp+simti
            if f==0:
                maxsim=sim
                id=tdata[1]
                f=1
            if f==1 and sim>maxsim:
                maxsim=sim
                id=tdata[1]
        else:
            None
    difference=maxsim
    print(simdtw,simA,simtp,simti)
    print(node.appliance_dict[event[0]],difference,id)
    node.appliance_dict[event[0]][1]=difference
    node.appliance_dict[event[0]][2]=id
    return True,difference,id

δ = lambda a,b: (a - b)**2
first = lambda x: x[0]
second = lambda x: x[1]

def calc_dtw(A, B):
    S = len(A)
    T = len(B)

    m = [[0 for j in range(T)] for i in range(S)]
    m[0][0] = (δ(A[0],B[0]), (-1,-1))
    for i in range(1,S):
        m[i][0] = (m[i-1][0][0] + δ(A[i], B[0]), (i-1,0))
    for j in range(1,T):
        m[0][j] = (m[0][j-1][0] + δ(A[0], B[j]), (0,j-1))

    for i in range(1,S):
        for j in range(1,T):
            minimum, index = minVal(m[i-1][j], m[i][j-1], m[i-1][j-1])
            indexes = [(i-1,j), (i,j-1), (i-1,j-1)]
            m[i][j] = (first(minimum)+δ(A[i], B[j]), indexes[index])
    return m

def minVal(v1, v2, v3):
    if first(v1) <= min(first(v2), first(v3)):
        return v1, 0
    elif first(v2) <= first(v3):
        return v2, 1
    else:
        return v3, 2
    
    
def calclate_likelihood(node):
    global d_list
    if node.delete!=1:
        d=0
        count=0
        for id in node.appliance_dict.keys():
            d=d+node.appliance_dict[id][1]
            count+=1
        if count==0:
            count=1
        d=d/count
        if 0 in d_list:
            d_list[d_list.index(0)]=d
        else:
            maxi=max(d_list)
            if maxi>d and node.delete!=1:
                d_list[d_list.index(maxi)]=d
    return 0

def check_dtw(node):
    global d_list
    if len(node.child_list)==0:        
        d=0
        count=0
        for id in node.appliance_dict.keys():
            d=d+node.appliance_dict[id][1]
            count+=1
        if count==0:
            count=1
        d=d/count
        maxi=max(d_list)
        if node.delete!=1 and maxi<d:
            node.delete=1      
        return 0
    for child in node.child_list:
        check_dtw(child)
    if len(node.child_list)==0:
        node.delete=1
    return 0

def prun_node(node):
    if len(node.child_list)==0:
        if node.delete==1:
            return False        
        return True
    newchild_list=[]
    for child in node.child_list:
        childflag=prun_node(child)
        if childflag:
            newchild_list.append(child)
    node.child_list=newchild_list
    if len(node.child_list)==0:
        return False
    return True
        
def print_leafnode(node):
    if len(node.child_list)==0:
        print(node.number)
    for child in node.child_list:
        print_leafnode(child)
    return 0
        
def delete_leaf(node):
    node.delete=1
    if len(node.parent.child_list)==0:
        delete_leaf(node.parent)
    return 0


def delete_duplication(node,valuelist): 
    if len(node.child_list)==0:
        d=0
        for id in node.appliance_dict.keys():
            d=d+node.appliance_dict[id][1]
        if d in valuelist:
            delete_leaf(node)
        else:
            valuelist.append(d)
        return valuelist
    for child in node.child_list:
        valuelist=delete_duplication(child,valuelist)
    return valuelist
    
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
class Treenode:
    def __init__(self, parent,id,appliance_dict,appliance,value,time,number):
        self.parent = parent
        
        self.id = id #次の家電にふるid
        
        self.appliance_dict = appliance_dict#今動いている家電とその[電力,相違度,推定される家電の種類,終了したか]
        
        self.appliance = appliance
        
        self.value = value
        
        self.time = time
        
        self.delete = 0
        
        self.number = number
        
        self.child_list = []
    
    

if __name__ == '__main__':
    #make_teacherdata()
    #estimate(make_smasockdatalist('2018-12-27-','2018-12-27-',0,3600),'2018-12-27-')
    #passlist=estimate(readdata(date,homenumber,0,4000),0)
    #estimate(filter(readdata(date,homenumber,0,3000)))
    #filter(readdata(date,homenumber,0,4000))
    #make_smasockdatalist("2018-12-30-","2018-12-31-",3200,4000)
    make_eventtree(make_smasockdatalist('2018-12-26-','2018-12-27-',800,1100),300,make_eventteacherdata())
    #make_eventteacherdata()
    
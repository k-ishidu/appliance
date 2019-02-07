#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:10:38 2018

@author: ishidukotaro
"""

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

def main():
    # クラスタ数
    N_CLUSTERS = 5

    # Blob データを生成する
    dataset = datasets.make_blobs(centers=N_CLUSTERS)

    # 特徴データ
    features = dataset[0]
    # 正解ラベルは使わない
    # targets = dataset[1]

    # クラスタリングする
    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)

    # 各要素をラベルごとに色付けして表示する
    for i in range(N_CLUSTERS):
        labels = features[pred == i]
        plt.scatter(labels[:, 0], labels[:, 1])

    # クラスタのセントロイド (重心) を描く
    centers = cls.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=100,
                facecolors='none', edgecolors='black')

    plt.show()


if __name__ == '__main__':
    main()
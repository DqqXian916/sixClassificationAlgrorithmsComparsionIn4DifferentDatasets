#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:14:36 2018
@author: dyx tc yxy
"""
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来在画图时正常显示中文标签

#读取数据集
def load_dataset(dataset_name,path,attribute_names):
    '''
    :param dataset_name: 数据集的名称
    :param path: 数据集的存放路径
    :param attribute_names: 数据集的属性列名
    :return: 返回读取到的数据集对象
    '''
    print("读取"+dataset_name+"数据集中...")
    dataset = pd.read_csv(path, names=attribute_names)
    print(dataset_name + "数据集读取完毕！")
    return dataset

#数据集的详细信息
def specification_dataset(dataset_name,dataset):
    '''
    :param dataset_name: 数据集名字
    :param dataset: 数据集对象
    :return: none
    '''
    print(dataset_name+"数据集的维度:", dataset.shape)
    print(dataset_name+"数据集的前3行:\n", dataset.head(3))  # 查看给定数据集前3行
    print("统计每种"+dataset_name+"的个数:")
    print(dataset.groupby('class').size())
    print(dataset_name+"数据集的描述统计信息(类型计数、平均值、标准差、最小值、四分位数、最大值等):")
    print(dataset.describe())

#画图可视化数据集
def visualize_dataset(dataset,dataset_name):
    '''
    :param dataset: 数据集对象
    :param dataset: 数据集名字
    :return:
    '''

    if dataset_name == "鸢尾花":
        dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
        dataset.plot(kind='hist', subplots=True, layout=(2, 2), sharex=False, sharey=False)
        colors = ['b', 'g', 'r']
        pd.plotting.scatter_matrix(dataset)
    elif dataset_name == "葡萄酒":
        # 盒图
        dataset.plot(kind='box', subplots=True, layout=(2,7),sharex=False, sharey=False,figsize=(14,7))
        # 直方图
        dataset.plot(kind='hist', subplots=True, layout=(7,2),sharex=False, sharey=False)
        print("箱图、直方图展示" + dataset_name + "数据集特征")
    elif dataset_name =="癌症患者生存期":
        dataset.plot(kind='box', subplots=True,sharex=False, sharey=False)
        dataset.plot(kind='hist', subplots=True,sharex=False, sharey=False)
        pd.plotting.scatter_matrix(dataset)
        print("箱图、直方图展示" + dataset_name + "数据集特征")
    elif dataset_name == "乳腺癌":
        dataset.plot(kind='box', subplots=True, sharex=False, sharey=False,figsize=(18,10))
        dataset.plot(kind='hist', subplots=True, sharex=False, sharey=False,figsize=(18,10))
        print("箱图、直方图展示" + dataset_name + "数据集特征")
    plt.show()

#评估六种分类算法
def six_classitifaction_algorithm_comparsion(dataset,dataset_name):
    '''
    :param dataset: 数据集对象
    :param dataset_name: 数据集名称
    :return:六种算法准确性评估结果的均值列表
    '''
    dataset_arr = dataset.values
    if dataset_name == "鸢尾花":
        dataset_attribute = dataset_arr[:, 0:4]  # 取前四列(鸢尾花的四种属性)
        dataset_type = dataset_arr[:,4]  # 取最后一列(取鸢尾花的类型)
    elif dataset_name == "葡萄酒":
        dataset_attribute = dataset_arr[:, 1:14]#取后十三列(葡萄酒的十三种属性)
        dataset_type = dataset_arr[:,0]#取第一列(取葡萄酒的类型)
        print(dataset_attribute.shape)
        print(dataset_type.shape)
    elif dataset_name == "癌症患者生存期":
        dataset_attribute = dataset_arr[:, 0:3]
        dataset_type = dataset_arr[:,3]
    elif dataset_name == "乳腺癌":
        dataset_attribute = dataset_arr[:,1:9]# C_D为编号，与Y无相关性，过滤掉
        dataset_type = dataset_arr[:,10]  # 取最后一列(取乳腺癌的类型)

    # 划分训练集和测试集，注意这里需要使用相同的随机种子，以确保每次运行时训练数据和测试数据是相同的
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataset_attribute, dataset_type, test_size=0.3,random_state=1)
    print("------------开始评估("+dataset_name+"数据集)------------")
    scoring = 'accuracy'
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDR', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    print("评估结果:")
    results = []
    names = []
    mean_accuracy = []#用于比较结果的最终accuracy
    head = "%s  %s  %s" % ("name", "mean", "std")
    print(head)
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=1)  # k折交叉验证
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        mean_accuracy.append(cv_results.mean())
        msg = "%s   %f  %f" % (name, cv_results.mean(), cv_results.std())
        print(msg)
   # 可视化展示比较结果
    fig = plt.figure()
    fig.suptitle(dataset_name + "数据集算法表现")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(dataset_name+".jpg")
    plt.show()
    print("------------结束评估(" + dataset_name + "数据集)------------")
    return mean_accuracy

#训练模型(包括了查看数据集的详细信息、可视化数据集以及算法的评估)
def train_model(attribute_names,dataset_name,dataset_obj):
    dataset_accuracy = []
    #查看数据集
    specification_dataset(dataset_name, dataset_obj)
    #可视化数据集
    visualize_dataset(dataset_obj,dataset_name)
    #在数据集上评估算法
    dataset_accuracy = six_classitifaction_algorithm_comparsion(dataset_obj,dataset_name)
    return dataset_accuracy

if __name__ == '__main__':
    iris_accuracy = []
    wine_accuracy = []
    haberman_accuracy = []
    bcw_accuracy = []
    ######################################鸢尾花(iris)数据集###############################################
    ##(三个臭皮匠)
    #读取鸢尾花数据集,并指定列名
    iris_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  # 指定鸢尾花的属性列名
    #加载鸢尾花数据集
    iris_dataset = load_dataset("鸢尾花", "dataset//iris.data", iris_names)
    #train_model包括了查看数据集的详细信息、可视化数据集以及算法的评估
    iris_accuracy = train_model(iris_names,"鸢尾花",iris_dataset)
    ######################################葡萄酒(wine)数据集###############################################
    ##(董怡馨)
    # 读取葡萄酒数据集,并指定列名
    wine_names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
    # 加载葡萄酒数据集
    wine_dataset = load_dataset("葡萄酒", "dataset//wine.data", wine_names)
    wine_accuracy = train_model(wine_names, "葡萄酒", wine_dataset)
    ######################################癌症患者生存期(haberman)数据集###############################################
    ##(唐楚)
    haberman_names = ['age', 'year', 'number', 'class']
    haberman_dataset = load_dataset("癌症患者生存期", "dataset//haberman.data", haberman_names)
    haberman_accuracy = train_model(haberman_names, "癌症患者生存期", haberman_dataset)
    ######################################乳腺癌(breast cancer)数据集#############################################
    ##(岳心怡)
    bcw_names = ['Sample code number', 'Clump Thickness', 'Uniformity od Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                 'Normal Nucleoli', 'Mitoses', 'class']
    bcw_dataset = load_dataset("乳腺癌", "dataset//breast-cancer-wisconsin.data", bcw_names)
    bcw_accuracy = train_model(bcw_names,"乳腺癌",bcw_dataset)
    ###可视化最终比较结果###
    x = range(0, 6, 1)
    algorithm_names = ['LR', 'LDR', 'KNN', 'NB', 'CART', 'SVM']
    plt.xticks(x, algorithm_names)
    plt.plot(x, iris_accuracy, color='red', label='iris')
    plt.plot(x, wine_accuracy, color='green', label='wine')
    plt.plot(x, haberman_accuracy, color='orange', label='haberman')
    plt.plot(x, bcw_accuracy, color='blue', label='breast cancer')

    plt.title('six classification algorithm in four different datasets')
    plt.xlabel('algorithm names')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('比较结果.jpg')
    plt.show()



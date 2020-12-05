from sklearn.metrics import precision_score,recall_score,roc_auc_score,roc_curve
from sklearn import tree
from sklearn.decomposition import PCA
import random
import pandas as pd
from sklearn.decomposition import NMF
import math
import numpy as np
import networkx as nx


def Getgauss_LncRNA(adjacentmatrix, nm):
    #       LncRNA Gaussian interaction profile kernels similarity
    KM = np.zeros((nm, nm))

    gamaa = 1
    sumnormm = 0
    for i in range(nm):
        normm = np.linalg.norm(adjacentmatrix[i]) ** 2
        sumnormm = sumnormm + normm
    gamam = gamaa / (sumnormm / nm)

    for i in range(nm):
        for j in range(nm):
            KM[i, j] = math.exp(-gamam * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2))
    return KM
def Getgauss_disease(adjacentmatrix,nd):
#Disease Gaussian interaction profile kernels similarity
    KD = np.zeros((nd,nd))
    gamaa=1
    sumnormd=0
    for i in range(nd):
        normd=(np.linalg.norm(adjacentmatrix[:,i])**2)
        sumnormd = sumnormd+normd
    gamad=gamaa/(sumnormd/nd)
    for i in range(nd):
        for j in range(nd):
            KD[i,j]= math.exp(-(gamad*(np.linalg.norm(adjacentmatrix[:,i]-adjacentmatrix[:,j])**2)))
    return KD


def threetypes_features(nm, nd, A, FS_integration, DS_integration):
    noOfObervationsOfLncRNA = np.zeros((nm, 1))  # number of observations in each row of MDA
    aveOfSimilaritiesOfLncRNA = np.zeros((nm, 1))  # average of all similarity scores for each LncRNA
    # histogram feature: cut [0, 1] into five bins and count the proportion of similarity scores that fall into each bin
    hist1LncRNA = np.zeros((nm, 1))
    hist2LncRNA = np.zeros((nm, 1))
    hist3LncRNA = np.zeros((nm, 1))
    hist4LncRNA = np.zeros((nm, 1))
    hist5LncRNA = np.zeros((nm, 1))

    for i in range(nm):
        noOfObervationsOfLncRNA[i, 0] = np.sum(A[i,])
        aveOfSimilaritiesOfLncRNA[i, 0] = np.mean(FS_integration[i,])
        # print (aveOfSimilaritiesOfLncRNA[i,0])
        hist1Count = 0.0
        hist2Count = 0.0
        hist3Count = 0.0
        hist4Count = 0.0
        hist5Count = 0.0
        for j in range(nm):
            if (FS_integration[i, j] < 0.2):
                hist1Count = hist1Count + 1.0
            elif (FS_integration[i, j] < 0.4):
                hist2Count = hist2Count + 1.0
            elif (FS_integration[i, j] < 0.6):
                hist3Count = hist3Count + 1.0
            elif (FS_integration[i, j] < 0.8):
                hist4Count = hist4Count + 1.0
            elif (FS_integration[i, j] <= 1):
                hist5Count = hist5Count + 1.0

        hist1LncRNA[i, 0] = hist1Count / nm
        hist2LncRNA[i, 0] = hist2Count / nm
        hist3LncRNA[i, 0] = hist3Count / nm
        hist4LncRNA[i, 0] = hist4Count / nm
        hist5LncRNA[i, 0] = hist5Count / nm

    # print (hist1LncRNA,hist2LncRNA,hist3LncRNA,hist4LncRNA,hist5LncRNA)
    feature1OfLncRNA = np.hstack(
        (noOfObervationsOfLncRNA, aveOfSimilaritiesOfLncRNA, hist1LncRNA, hist2LncRNA, hist3LncRNA, hist4LncRNA, hist5LncRNA))
    # print ('feature1OfLncRNA',feature1OfLncRNA[0])
    ################################
    ## Type 1 feature of diseases ##
    ################################

    noOfObervationsOfdisease = np.zeros((nd, 1))  # number of observations in each column of MDA
    aveOfsimilaritiesOfDisease = np.zeros((nd, 1))  # average of all similarity scores for each disease
    hist1disease = np.zeros((nd,
                             1))  # histogram feature: cut [0, 1] into five bins and count the proportion of similarity scores that fall into each bin
    hist2disease = np.zeros((nd, 1))
    hist3disease = np.zeros((nd, 1))
    hist4disease = np.zeros((nd, 1))
    hist5disease = np.zeros((nd, 1))
    for i in range(nd):
        noOfObervationsOfdisease[i, 0] = np.sum(A[:, i])
        aveOfsimilaritiesOfDisease[i] = np.mean(DS_integration[i])
        hist1Count = 0.0
        hist2Count = 0.0
        hist3Count = 0.0
        hist4Count = 0.0
        hist5Count = 0.0
        for j in range(nd):
            if (DS_integration[i, j] < 0.2):
                hist1Count = hist1Count + 1.0
            elif (DS_integration[i, j] < 0.4):
                hist2Count = hist2Count + 1.0
            elif (DS_integration[i, j] < 0.6):
                hist3Count = hist3Count + 1.0
            elif (DS_integration[i, j] < 0.8):
                hist4Count = hist4Count + 1.0
            elif (DS_integration[i, j] <= 1):
                hist5Count = hist5Count + 1.0

        hist1disease[i, 0] = hist1Count / nd
        hist2disease[i, 0] = hist2Count / nd
        hist3disease[i, 0] = hist3Count / nd
        hist4disease[i, 0] = hist4Count / nd
        hist5disease[i, 0] = hist5Count / nd

    feature1OfDisease = np.hstack((noOfObervationsOfdisease, aveOfsimilaritiesOfDisease, hist1disease, hist2disease,
                                   hist3disease, hist4disease, hist5disease))
    # print ('feature1OfDisease',feature1OfDisease[0])
    #############################
    # Type 2 feature of LncRNAs ##
    #############################

    # number of neighbors of LncRNAs and similarity values for 10 nearest neighbors
    numberOfNeighborsLncRNA = np.zeros((nm, 1))
    similarities10KnnLncRNA = np.zeros((nm, 10))
    averageOfFeature1LncRNA = np.zeros((nm, 7))
    weightedAverageOfFeature1LncRNA = np.zeros((nm, 7))
    similarityGraphLncRNA = np.zeros((nm, nm))
    meanSimilarityLncRNA = np.mean(FS_integration)
    for i in range(nm):
        neighborCount = 0 - 1  # similarity between an LncRNA and itself is not counted
        for j in range(nm):
            if (FS_integration[i, j] >= meanSimilarityLncRNA):
                neighborCount = neighborCount + 1
                similarityGraphLncRNA[i, j] = 1
        numberOfNeighborsLncRNA[i, 0] = neighborCount

        similarities10KnnLncRNA[i,] = sorted(FS_integration[i,], reverse=True)[1:11]
        indices = np.argsort(-FS_integration[i,])[1:11]

        averageOfFeature1LncRNA[i,] = np.mean(feature1OfLncRNA[indices,], 0)
        weightedAverageOfFeature1LncRNA[i,] = np.dot(similarities10KnnLncRNA[i,], feature1OfLncRNA[indices,]) / 10
        # build LncRNA similarity graph
    mSGraph = nx.from_numpy_matrix(similarityGraphLncRNA)
    betweennessCentralityLncRNA = np.array(list(nx.betweenness_centrality(mSGraph).values())).T
    # print ("numberOfNeighborsLncRNA",numberOfNeighborsLncRNA[0,0],'similarities10KnnLncRNA',similarities10KnnLncRNA[0])#betweennessCentralityLncRNA.shape
    # print (betweennessCentralityLncRNA)
    # print (np.array(betweennessCentralityLncRNA.values()))
    # closeness_centrality
    closenessCentralityLncRNA = np.array(list(nx.closeness_centrality(mSGraph).values())).T
    # print (closenessCentralityLncRNA.shape)
    # pagerank
    pageRankLncRNA = np.array(list(nx.pagerank(mSGraph).values())).T
    # print (pageRankLncRNA.shape)
    # eigenvector_centrality
    # eigenvector_centrality=nx.eigenvector_centrality(mSGraph)
    eigenVectorCentralityLncRNA = np.array(list(nx.eigenvector_centrality(mSGraph).values())).T
    # print (eigenVectorCentralityLncRNA.shape)
    combination = np.array(
        [betweennessCentralityLncRNA, closenessCentralityLncRNA, pageRankLncRNA, eigenVectorCentralityLncRNA])
    # print (combination)
    # print (combination.shape)
    # # concatenation
    feature2OfLncRNA = np.hstack((numberOfNeighborsLncRNA, similarities10KnnLncRNA, averageOfFeature1LncRNA,
                                 weightedAverageOfFeature1LncRNA,
                                 combination.T))  # betweennessCentralityLncRNA, closenessCentralityLncRNA, eigenVectorCentralityLncRNA, pageRankLncRNA))
    # print ('feature2OfLncRNA',feature2OfLncRNA[0])
    ###############################
    # Type 2 feature of diseases ##
    ###############################

    # number of neighbors of diseases and similarity values for 10 nearest neighbors
    numberOfNeighborsDisease = np.zeros((nd, 1))
    similarities10KnnDisease = np.zeros((nd, 10))
    averageOfFeature1Disease = np.zeros((nd, 7))
    weightedAverageOfFeature1Disease = np.zeros((nd, 7))
    similarityGraphDisease = np.zeros((nd, nd))
    meanSimilarityDisease = np.mean(DS_integration)
    for i in range(nd):
        neighborCount = 0 - 1
        for j in range(nd):
            if (DS_integration[i, j] >= meanSimilarityDisease):
                neighborCount = neighborCount + 1
                similarityGraphDisease[i, j] = 1

        numberOfNeighborsDisease[i, 0] = neighborCount

        similarities10KnnDisease[i,] = sorted(DS_integration[i,], reverse=True)[1:11]
        indices = np.argsort(-DS_integration[i,])[1:11]

        averageOfFeature1Disease[i,] = np.mean(feature1OfDisease[indices,], 0)
        weightedAverageOfFeature1Disease[i,] = np.dot(similarities10KnnDisease[i,], feature1OfDisease[indices,]) / 10

    # build disease similarity graph
    dSGraph = nx.from_numpy_matrix(similarityGraphDisease)
    betweennessCentralityDisease = np.array(list(nx.betweenness_centrality(dSGraph).values())).T
    # print (betweenness_centrality)
    # closeness_centrality
    closenessCentralityDisease = np.array(list(nx.closeness_centrality(dSGraph).values())).T
    # print (closeness_centrality)
    # pagerank
    pageRankDisease = np.array(list(nx.pagerank(dSGraph).values())).T
    # print (pagerank)
    # eigenvector_centrality
    eigenVectorCentralityDisease = np.array(list(nx.eigenvector_centrality(dSGraph).values())).T
    # print (eigenvector_centrality)
    combination = np.array(
        [betweennessCentralityDisease, closenessCentralityDisease, pageRankDisease, eigenVectorCentralityDisease])
    # print (combination)
    # print (combination.shape)

    # concatenation
    feature2OfDisease = np.hstack((numberOfNeighborsDisease, similarities10KnnDisease, averageOfFeature1Disease,
                                   weightedAverageOfFeature1Disease,
                                   combination.T))  # betweennessCentralityDisease, closenessCentralityDisease, eigenVectorCentralityDisease, pageRankDisease))
    # print ('feature2OfDisease',feature2OfDisease[0])
    ###########################################
    ## Type 3 feature of LncRNA-disease pairs ##
    ###########################################

    # matrix factorization
    # number of associations between an LncRNA and a disease's neighbors
    nmf_model = NMF(n_components=20)
    latentVectorsLncRNA = nmf_model.fit_transform(A)
    latentVectorsDisease = nmf_model.components_
    numberOfDiseaseNeighborAssociations = np.zeros((nm, nd))
    numberOfLncRNANeighborAssociations = np.zeros((nm, nd))
    MDAGraph = nx.Graph()
    MDAGraph.add_nodes_from(list(range(nm + nd)))
    for i in range(nm):
        for j in range(nd):
            if A[i, j] == 1:
                MDAGraph.add_edge(i, j + nm)  # build MDA graph
            for k in range(nd):
                if DS_integration[j, k] >= meanSimilarityDisease and A[i, k] == 1:
                    numberOfDiseaseNeighborAssociations[i, j] = numberOfDiseaseNeighborAssociations[i, j] + 1

            for l in range(nm):
                if FS_integration[i, l] >= meanSimilarityLncRNA and A[l, j] == 1:
                    numberOfLncRNANeighborAssociations[i, j] = numberOfLncRNANeighborAssociations[i, j] + 1

    # betweennessCentralityMDA=nx.betweenness_centrality(MDAGraph)
    betweennessCentralityMDA = np.array(list(nx.betweenness_centrality(MDAGraph).values())).T
    betweennessCentralityLncRNAInMDA = betweennessCentralityMDA[0:nm]
    betweennessCentralityDiseaseInMDA = betweennessCentralityMDA[nm:nm+nd]
    # print (betweenness_centrality)
    closenessCentralityMDA = np.array(list(nx.closeness_centrality(MDAGraph).values())).T
    closenessCentralityLncRNAInMDA = closenessCentralityMDA[0:nm]
    closenessCentralityDiseaseInMDA = closenessCentralityMDA[nm:nm+nd]
    eigenVectorCentralityMDA = np.array(
        list(nx.eigenvector_centrality_numpy(MDAGraph).values())).T  # nx.eigenvector_centrality(MDAGraph)
    eigenVectorCentralityLncRNAInMDA = eigenVectorCentralityMDA[0:nm]
    eigenVectorCentralityDiseaseInMDA = eigenVectorCentralityMDA[nm:nd+nm]
    pageRankMDA = np.array(list(nx.pagerank(MDAGraph).values())).T  # nx.pagerank(MDAGraph)
    pageRankLncRNAInMDA = pageRankMDA[0:nm]
    pageRankDiseaseInMDA = pageRankMDA[nm:nm+nd]

    Diseasecombination = np.array(
        [betweennessCentralityDiseaseInMDA, closenessCentralityDiseaseInMDA, eigenVectorCentralityDiseaseInMDA,
         pageRankDiseaseInMDA])
    feature3OfDisease = np.hstack((latentVectorsDisease.T, Diseasecombination.T))
    LncRNAcombination = np.array(
        [betweennessCentralityLncRNAInMDA, closenessCentralityLncRNAInMDA, eigenVectorCentralityLncRNAInMDA,
         pageRankLncRNAInMDA])
    feature3OfLncRNA = np.hstack((latentVectorsLncRNA, LncRNAcombination.T))
    # print ('feature3OfLncRNA',feature3OfLncRNA[0])
    # print ('feature3OfDisease',feature3OfDisease[0])
    Feature_LncRNA = np.hstack((feature1OfLncRNA, feature2OfLncRNA, feature3OfLncRNA))
    Feature_disease = np.hstack((feature1OfDisease, feature2OfDisease, feature3OfDisease))
    return Feature_LncRNA, Feature_disease, numberOfDiseaseNeighborAssociations, numberOfLncRNANeighborAssociations
def getdata1():
    DS2 = np.loadtxt('LL1.txt') # lncRNA similarity matrix
    return DS2
def getdata3():
    DS1 = np.loadtxt('LD1.txt')# disease similarity matrix
    return DS1
def getdata2():
    DS3 = np.loadtxt('knowndiseaselncrnainteraction.txt')
    DS3 = DS3.T
    z = np.zeros((nd, nm))  # disease and lncRNA interaction 0-1 matrix
    for i in range(293):
        z[int(DS3[0][i]) - 1][int(DS3[1][i]) - 1] = 1
    z = z.T
    return z
data = getdata1()
data2 = getdata3()
DS = data2
FS = data
nm=118
nd=167
FSweight=np.zeros((nm,nm))
for i in range(nm):
    for j in range(nm):
        if(FS[i][j]==0):FSweight[i][j]=0
        else:FSweight[i][j]=1

DSweight=np.zeros((nd,nd))
for i in range(nd):
    for j in range(nd):
        if(DS[i][j]==0):DSweight[i][j]=0
        else:DSweight[i][j]=1
Ar=np.zeros((nm,nd))


def run(a, b):
    # 数据输入
    data1 = getdata2()
    xz = np.array(data1, dtype=np.int)
    #     xz=np.array(data1,dtype=np.float32)

    if (xz[a][b] == 1): xz[a][b] = 0
    pca = PCA(n_components=10)
    nc = xz.sum()

    nm = 118  # number of LncRNAs
    nd = 167  # number of diseases
    #     nc = 539 # number of LncRNA-disease associations
    r = 0.5 # Decising the size of feature subset
    nn = nm * nd - nc  # number of unknown samples
    M = 50  # number of decison trees
    # A = np.zeros((nm,nd),dtype=float)
    # ConnectDate = np.loadtxt(r'.\data\known disease-LncRNA association number ID.txt',dtype=int)-1
    A = xz
    # for i in range(nc):
    #     A[ConnectDate[i,0], ConnectDate[i,1]] = 1 # the element is 1 if the LncRNA-disease pair has association
    dataset_n = np.argwhere(A == 0)
    Trainset_p = np.argwhere(A == 1)
    KM = Getgauss_LncRNA(A, nm)
    KD = Getgauss_disease(A, nd)

    # integrating LncRNA functional similarity and Gaussian interaction profile kernels similarity
    FS_integration = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            if (FSweight[i, j] == 1):
                #         if  FSweight[i,j] == 1:
                FS_integration[i, j] = FS[i, j]
            else:
                FS_integration[i, j] = KM[i, j]
    DS_integration = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):

            #         if  DS[i,j] >= q2 :
            if (DSweight[i, j] == 1):
                DS_integration[i, j] = DS[i, j]

            else:
                DS_integration[i, j] = KD[i, j]
    LncrnaFeature, DiseaseFeature, numberOfDiseaseNeighborAssociations, \
    numberOfLncRNANeighborAssociations = threetypes_features(nm, nd, A, FS_integration, DS_integration)

    predict_0 = np.zeros((dataset_n.shape[0]))
    for i_M in range(M):
        Trainset_n = dataset_n[random.sample(list(range(nn)), nc)]

        # print (Trainset_n)
        Trainset = np.vstack((Trainset_n, Trainset_p))

        TrainLncrnaFeature = LncrnaFeature[Trainset[:, 0]]
        TrainDiseaseFeature = DiseaseFeature[Trainset[:, 1]]

        LncrnaNumberNeighborTrain = numberOfLncRNANeighborAssociations[Trainset[:, 0], Trainset[:, 1]]
        DiseaseNumberNeighborTrain = numberOfDiseaseNeighborAssociations[Trainset[:, 0], Trainset[:, 1]]

        TrainLncrnaFeatureOfPair = np.hstack(
            (TrainLncrnaFeature, DiseaseNumberNeighborTrain.reshape(DiseaseNumberNeighborTrain.shape[0], 1)))
        randomNum_LncrnaFeature = random.sample(list(range(TrainLncrnaFeatureOfPair.shape[1])),
                                               int(r * TrainLncrnaFeatureOfPair.shape[1]))
        TrainLncrnaFeatureOfPair_random = TrainLncrnaFeatureOfPair[:, randomNum_LncrnaFeature]
        PCA_TrainLncrnaFeatureOfPair = pca.fit_transform(TrainLncrnaFeatureOfPair_random)
        PCA_LncRNATrainVarianceRatio = pca.explained_variance_ratio_
        TrainDiseaseFeatureOfPair = np.hstack(
            (TrainDiseaseFeature, LncrnaNumberNeighborTrain.reshape(LncrnaNumberNeighborTrain.shape[0], 1)))
        randomNum_diseaseFeature = random.sample(list(range(TrainDiseaseFeatureOfPair.shape[1])),
                                                 int(r * TrainDiseaseFeatureOfPair.shape[1]))
        TrainDiseaseFeatureOfPair_random = TrainDiseaseFeatureOfPair[:, randomNum_diseaseFeature]
        PCA_TrainDiseaseFeatureOfPair = pca.transform(TrainDiseaseFeatureOfPair_random)
        PCA_diseaseTrainVarianceRatio = pca.explained_variance_ratio_

        X_train = np.hstack((PCA_TrainLncrnaFeatureOfPair, PCA_TrainDiseaseFeatureOfPair))

        Y_value = []
        for i in range(Trainset_n.shape[0]):
            Y_value.append(0.0)
        for i in range(Trainset_n.shape[0], Trainset.shape[0]):
            Y_value.append(1.0)

        clf = tree.DecisionTreeRegressor(splitter='random', min_samples_split=3, min_samples_leaf=2)
        clf = clf.fit(X_train, Y_value)

        TestLncrnaFeature = LncrnaFeature[dataset_n[:, 0]]
        TestDiseaseFeature = DiseaseFeature[dataset_n[:, 1]]

        LncrnaNumberNeighborTest = numberOfLncRNANeighborAssociations[dataset_n[:, 0], dataset_n[:, 1]]
        DiseaseNumberNeighborTest = numberOfDiseaseNeighborAssociations[dataset_n[:, 0], dataset_n[:, 1]]

        TestLncrnaFeatureOfPair = np.hstack(
            (TestLncrnaFeature, DiseaseNumberNeighborTest.reshape(DiseaseNumberNeighborTest.shape[0], 1)))
        TestLncrnaFeatureOfPair_random = TestLncrnaFeatureOfPair[:, randomNum_LncrnaFeature]
        PCA_TestLncrnaFeatureOfPair = pca.transform(TestLncrnaFeatureOfPair_random)
        PCA_LncRNATestVarianceRatio = pca.explained_variance_ratio_
        TestDiseaseFeatureOfPair = np.hstack(
            (TestDiseaseFeature, LncrnaNumberNeighborTest.reshape(LncrnaNumberNeighborTest.shape[0], 1)))
        TestDiseaseFeatureOfPair_random = TestDiseaseFeatureOfPair[:, randomNum_diseaseFeature]
        PCA_TestDiseaseFeatureOfPair = pca.transform(TestDiseaseFeatureOfPair_random)
        PCA_diseaseTestVarianceRatio = pca.explained_variance_ratio_

        # X_test = np.hstack((FS_test,DS_test,LncRNADiseaseFeatureTest))
        X_test = np.hstack((PCA_TestLncrnaFeatureOfPair, PCA_TestDiseaseFeatureOfPair))

        predict_0 = predict_0 + clf.predict(X_test)  # prediction results of all unknown samples

    predict_0 = predict_0 / M

    predict_0scoreranknumber = np.argsort(-predict_0)
    predict_0scorerank = predict_0[predict_0scoreranknumber]
    diseaserankname_pos = dataset_n[predict_0scoreranknumber, 1]
    LncRNArankname_pos = dataset_n[predict_0scoreranknumber, 0]
    prediction_0_out = [diseaserankname_pos, LncRNArankname_pos, predict_0scorerank]

    f = np.vstack((diseaserankname_pos, LncRNArankname_pos, predict_0scorerank))
    c = f.T
    Ac = np.zeros((nm, nd), dtype=float)
    Ac = Ac
    for i in range(nn):
        Ac[int(c[i, 1]), int(c[i, 0])] = c[i, 2]  # the element is 1 if the LncRNA-disease pair has association
    Ar[a][b] = Ac[a][b]
for i in range(nm):
    for j in range(nd):
        run(i,j)
        np.savetxt(r'118-167Loocv.txt',
                   Ar, delimiter=',')
        print((i*nd+j)/(nm*nd))
data1 = getdata2()
xz=np.array(data1,dtype=np.int)
d11=xz.ravel()
d12=Ar.ravel()
auc=roc_auc_score(d11,d12)
print(auc)
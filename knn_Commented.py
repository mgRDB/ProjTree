"""
kth Nearest Neighbor Projection Forest
November 2023
@author: Ryan Berry
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


####################################
#### Tree Creation & Navigation ####
####################################


class Queue:

    """
    Used for organizing nodes and splitting nodes
    """

    def __init__(self, l):
        self.q = []
        self.q.extend(l)
        
    def is_empty(self):
        return len(self.q) == 0
    
    def size(self):
        return len(self.q)
    
    def enqueue(self,val):
        self.q.append(val)
        
    def dequeue(self):
        return self.q.pop(0)



class Node:

    """
    Tree Structure that stores information for each tree in the forest alongside a dataframe
    """

    def __init__(self,table):
        self.t = table
        self.l = None
        self.r = None  
        
    @property
    def left(self):
        return self.l
    
    @left.setter
    def left(self,left):
        self.l = left
        
    @property
    def right(self):
        return self.r
    
    @right.setter
    def right(self,right):
        self.r = right



def project(node, splits, location = 0):
    
    """

    Projects given data onto multiple possible lines, returning the line with the most diversity in spread

    Parameters
    ----------
    node : the node that contains the current data being worked on
    splits : the number of projections to test before choosing the most effective one
    location : where along the projection to split the data
        0 is the median and the default
        1 is the largest gap between 10 uniformly generated points
        anything else is the largest gap between all projections

    Returns
    -------
    left : a node that contains all data smaller than the split point
    right : a node that contains all dat larger than or equal to the split point

    """
    
    #Generate a random projection to split the data on
    sdProj = -1
    
    for i in range(splits):
        
        theta = np.random.normal(0,1,len(node.t.columns))
        temp = node.t.multiply(theta).sum(axis=1)
        
        if temp.std() > sdProj:
            
            proj = temp
            sdProj = temp.std()
            best = theta
        
    #Choose a point along the projection to split the data
    
    #Split at the median of the projection
    if location == 0:
        
        s = proj.median()
        
    #Split at the largest gap between uniformly generated points across the projection
    elif location == 1:
        
        ns = 10
        ss = np.random.uniform(low = min(proj), high = max(proj), size = ns)
        gap = 0
        
        for j in range(len(ss)):
            
            temp = proj - ss[j]
            
            if min(temp[temp>0]) - max(temp[temp<=0])>gap:
                
                gap = min(temp[temp>0]) - max(temp[temp<=0])
                ind = j
                
        s = ss[ind]
        
    #Split at the largest gap between all points in the projection
    else:
        
        pr = proj.sort_values().reset_index(drop=True)
        p1 = pr.drop(0).reset_index(drop=True)
        p2 = pr.drop(len(proj)-1)
        p = p1 - p2
        s = pr[p.argmin()+1]
    l = node.t[proj<s]
    r = node.t[proj>s]
    m = node.t[proj==s]
    k = (len(m)//2)
    if len(r) > len(l):
        l = pd.concat([l, m[k:]])
        r = pd.concat([r, m[:k]])
    else:
        l = pd.concat([l, m[:k]])
        r = pd.concat([r, m[k:]])
    if True:
        l = node.t[proj<s]
        r = node.t[proj>=s]
    left = Node(l)
    right = Node(r)
    return left, right, best, s



def build_tree(data,leaf,splits,location = 0):
    
    """

    Creates a tree of the data split into different partions through projections

    Parameters
    ----------
    data : pandas dataframe that is being made into a tree
    leaf : maximum size that the leaf nodes can be
    splits : number of projections to try each time a node is split
    location : where along the projection to split the data
        0 is the median and the default
        1 is the largest gap between 10 uniformly generated points
        anything else is the largest gap between all projections

    Returns
    -------
    root : the first node in the created tree
    table : a dataframe of each node, its children and the number of data points it contains

    """
    
    #Inatiate the root node and the dataframe for recording nodes
    table = pd.DataFrame({"Node":[],"Left":[],"Right":[],"Size":[],"Theta":[],"Median":[]}, dtype=object)
    root = Node(data)
    order = Queue([root])
#    i = 0
    
    #Start a queue of nodes to be split
    while(not order.is_empty()):
        
        current = order.dequeue()
        
        #Split nodes that are larger than the maximum leaf size and add their children to the queue
        if len(current.t) > leaf:
            
            current.l, current.r, theta, median = project(current,splits, location)
            order.enqueue(current.l)
            order.enqueue(current.r)
            table.loc[len(table.index)] = [current, current.l, current.r, len(current.t.index), theta, median]
            
        #Remove nodes that are smaller than the maximum leaf size from the queue and record them as a leaf in the table
        else:
            
            table.loc[len(table.index)] = [current, np.nan, np.nan, len(current.t.index), np.nan, np.nan]
    return root, table



def traversal (row, root, table):

    """

    Locates the leaf node that a row would be in given a tree

    Parameters
    ----------
    row : an data entry matching the data format of the tree being traversed
    root : root node of the tree to be traversed
    table : dataframe containing every node of a created tree

    Returns
    -------
    node : node object pointing to a leaf in the tree that would contain the row input

    """

    #Find where the given root is located
    for i in range(len(table)):
        if table["Node"][i] is root:
            temp = i
            break

    #Check if the current node is a leaf and return if true
    if np.any(np.isnan(table["Theta"][temp])):
        return table["Node"][temp]
    
    #Move to next child and repeat
    proj = row.multiply(table["Theta"][temp]).sum()
    if proj >= table["Median"][temp]:
        return traversal(row, table["Right"][temp], table)
    else:
        return traversal(row, table["Left"][temp], table)



def planter (data, trees, leafs, splits):

    """

    Builds a forest by creating multiple trees and storing them in a dataframe

    Parameters
    ----------
    data : numerical data to be turned into a forest
    trees : number of trees to be created
    leafs : maximum number of entries that a leaf in a tree can contain
    splits : number of different projections tried each time a node is split

    Returns
    -------
    forest : a dataframe containing the root of each tree and a table of all of their nodes

    """

    forest = pd.DataFrame({"Root":[],"Table":[]}, dtype=object)
    for i in range(trees):
        root, table = build_tree(data, leafs, splits)
        forest.loc[len(forest.index)] = [root, table]
    return forest



def finder (row, forest):

    """

    Finds the union of all leaf nodes in a forest for a given data entry

    Parameters
    ----------
    row : data entry used for traversal to find which leaf it resides in
    forest : dataframe containing the root and nodes of each tree in a forest

    Returns
    -------
    result : a dataframe containing the union of the leaf node reached from traversing each tree

    """

    #Traverse each tree in the forest and concatenate the leaf nodes
    r = len(forest.index)
    result = traversal(row, forest["Root"][0], forest["Table"][0]).t
    if r > 1:
        for i in range(1, r):
            temp = traversal(row, forest["Root"][i], forest["Table"][i]).t
            result = pd.concat([result, temp])

    #Remove duplicate entries and return dataframe
    result = result[~result.index.duplicated(keep="first")]
    return result


#############################
#### k Nearest Neighbors ####
#############################


def closest (row, lot, k):

    """

    Finds the closest k entries in the grouping of all close entries

    Parameters
    ----------
    row : data entry used for traversal and distance calculation
    lot : union of leafs containing closest entries
    k : number of entries to be returned

    Returns
    -------
    nearest : a dataframe containing the k closest entries

    """

    #remove the entry itself from being considered
    if row.name in lot.index:
        top = lot.drop([row.name])
    else:
        top = lot
    
    #record distances to each entry and return the top k
    top["remove"] = 0
    for i in range(len(top.columns)-1):
        top['remove'] = top["remove"] + (top[top.columns[i]] - row[i])**2
    top["remove"] = np.sqrt(top["remove"])
    top = top.sort_values(by=["remove"])
    return top.loc[:, top.columns != 'remove'].iloc[:k]



def correct (whole, partial):

    """
    Used for determining how many predictions matched the true values
    """

    return sum(el in whole.index for el in partial.index)



def accuracy (data, forest, k):

    """
    Accuracy calculation of predictions
    """

    n = len(data)
    sum = 0
    for i in range(n):
        w = closest(data.iloc[i],data,k)
        p = closest(data.iloc[i],finder(data.iloc[i], forest),k)
        sum += correct(w,p)
    return sum / (n * k)



def acctable (data, trees, leaves, splits, ks):

    """
    Record of accuracy calculations with given parameters 
    """

    table = pd.DataFrame({"Tree Count":[],"Leaf Count":[],"Split Count":[],"K Count":[],"Accuracy":[]})
    i = 0
    t = len(trees)*len(leaves)*len(splits)*len(ks)
    for tree in trees:
        for leaf in leaves:
            for split in splits:
                for k in ks:
                    table.loc[len(table.index)] = [tree, leaf, split, k, accuracy(data, planter(data, tree, leaf, split), k)]
                    i += 1
                    print("Loop {}: {:.2f}% Done".format(i, 100*i/t))
    return table



def knn (data, index, forest, ks):

    """
    Easier way of calling the needed functions
    """

    return closest(data.iloc[index], finder(data.iloc[index], forest), ks)



def normalized_planter (data, trees, leafs, splits):

    """
    Planter but it normalizes the data first
    """

    normal = (data - data.mean())/data.std()
    return planter(normal, trees, leafs, splits)



def normalized_knn (data, index, forest, ks):

    """
    knn but for used with normalized forests
    """

    normal = (data - data.mean())/data.std()
    return data.iloc[knn(normal, index, forest, ks).index]



def closest_with_distance (row, lot, k):

    """
    Closest but the calculated distances are retained upon returning
    """

    if row.name in lot.index:
        top = lot.drop([row.name])
    else:
        top = lot
    top["remove"] = 0
    for i in range(len(top.columns)-1):
        top["remove"] = top["remove"] + (top[top.columns[i]] - row[i])**2
    top["remove"] = np.sqrt(top["remove"])
    top = top.sort_values(by=["remove"])
    return top.iloc[:k]



def anomaly_detector (distance, threshhold):

    """
    Basic anomaly classification using a given threshhold value
    """

    if distance.mean() > threshhold:
        return True
    else:
        return False


#############################
#### Test Point Creation ####
#############################


def anom_between(std1, mean1, std2, mean2):

    """
    point inbetween two clusters based on their standard deviations
    """

    vector = mean2-mean1
    vector = 1*vector/abs(vector).max()
    std1 = std1*vector
    std2 = std2*vector
    anomaly = mean1 + ((std1*(mean2-mean1))/(std1+std2))
    return anomaly



def non_anom_farin(std1, mean1, mean2):

    """
    random points close to the inner edge of the cluster in the "positive" direction
    """

    vector = mean2-mean1
    vector = 1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(0,1)+2)
    return anomaly



def non_anom_inner(std1, mean1, mean2):

    """
    random points close to the center of the cluster in the "positive" direction
    """

    vector = mean2-mean1
    vector = 1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(0,2))
    return anomaly



def anom_inner(std1, mean1, mean2):

    """
    random points close to the outer edge of the cluster in the "positive" direction
    """

    vector = mean2-mean1
    vector = 1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(0,1)+3)
    return anomaly



def anom_farin(std1, mean1, mean2):

    """
    random points far from the outer edge of the cluster in the "positive" direction
    """

    vector = mean2-mean1
    vector = 1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(1,2)+3)
    return anomaly



def non_anom_farout(std1, mean1, mean2):

    """
    random points close to the inner edge of the cluster in the "negative" direction
    """

    vector = mean2-mean1
    vector = -1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(0,1)+2)
    return anomaly



def non_anom_outer(std1, mean1, mean2):

    """
    random points close to the center of the cluster in the "negative" direction
    """

    vector = mean2-mean1
    vector = -1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(0,2))
    return anomaly



def anom_outer(std1, mean1, mean2):

    """
    random points close to the outer edge of the cluster in the "negative" direction
    """

    vector = mean2-mean1
    vector = -1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(0,1)+3)
    return anomaly



def anom_farout(std1, mean1, mean2):

    """
    random points far from the outer edge of the cluster in the "negative" direction
    """

    vector = mean2-mean1
    vector = -1*vector/abs(vector).max()
    std1 = std1*vector
    anomaly = mean1 + std1*(np.random.uniform(1,2)+3)
    return anomaly


########################
#### Importing Data ####
########################


#### Parameters ####

#Number of closest points
k = 3

#Number of projections
tree = 5

#Leaf node maximum size
leaf = 5

#Number of projections tried
split = 5


#### Importing ####

#Iris Data
# iris = fetch_ucirepo(id=53)
# test = iris.data.features

#Cancer Data
cancer = fetch_ucirepo(id=17)
test = cancer.data.features

#Glass Data
# glass = fetch_ucirepo(id=42)
# test = glass.data.features

#Lung Data
#lung = fetch_ucirepo(id=62)
#test = lung.data.features

#Wholesale Data
#wholesale = fetch_ucirepo(id=292)
#test = wholesale.data.features

#Yeast Data
# yeast = fetch_ucirepo(id=110)
# test = yeast.data.features

#Parkinsons
# park = fetch_ucirepo(id=174)
# test = park.data.features

#Account for duplicate column names
test.columns = range(len(test.columns))

# targ = iris.data.targets
# test["class"] = targ

#For more specific test point creation
# std1 = test[test["class"]=='Iris-setosa'].drop(columns=["class"]).std()
# mean1 = test[test["class"]=='Iris-setosa'].drop(columns=["class"]).mean()
# std2 = test[test["class"]!='Iris-setosa'].drop(columns=["class"]).std()
# mean2 = test[test["class"]!='Iris-setosa'].drop(columns=["class"]).mean()



targ = cancer.data.targets
test["class"] = targ


# targ = glass.data.targets
# test["class"] = targ


# targ = lung.data.targets
# test["class"] = targ


# targ = park.data.targets
# test["class"] = targ


# targ = yeast.data.targets
# test["class"] = targ


testless = test.drop(columns = ["class"])


test.fillna(method="ffill",inplace=True)
test.fillna(method="bfill",inplace=True)

std1 = test.drop(columns=["class"]).std()
mean1 = test.drop(columns=["class"]).mean()
mean2 = test.drop(columns=["class"]).iloc[0]

normal = (test - test.mean())/test.std()
n = len(normal)


##############################################
#### Generating Possible "Past Anomalies" ####
##############################################


#### Making Possible Entries ####

know = pd.DataFrame(columns = test.columns)

for i in range(20):
    # Inner Points Within The Cluster Close To The Mean
    anomaly = non_anom_inner(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Inner Points Within The Cluster Far From The Mean
    anomaly = non_anom_farin(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Inner Points Far From The Cluster
    anomaly = anom_farin(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Inner Points Close To The Cluster
    anomaly = anom_inner(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Outer Points Within The Cluster Close To The Mean
    anomaly = non_anom_outer(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Outer Points Within The Cluster Far From The Mean
    anomaly = non_anom_farout(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Outer Points Close To The Cluster
    anomaly = anom_outer(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    
    # Outer Point Far From The Cluster
    anomaly = anom_farout(std1, mean1, mean2)
    anomaly["class"] = "anomaly"
    know.loc[len(know.index)] = anomaly
    

#### Classify Possible Entries ####

true = []

# Breast Cancer

B = test[test["class"]=='B'].drop(columns=["class"])
Bm = B.quantile(0.5)
Bd = B - Bm
Bd = Bd**2
Bd = Bd.sum(axis=1)
Bd = np.sqrt(Bd)
Bdm = Bd.quantile(0.5)
Bdiqr = Bd.quantile(0.75)-Bd.quantile(0.25)

M = test[test["class"]=='M'].drop(columns=["class"])
Mm = M.quantile(0.5)
Md = M - Mm
Md = Md**2
Md = Md.sum(axis=1)
Md = np.sqrt(Md)
Mdm = Md.quantile(0.5)
Mdiqr = Md.quantile(0.75)-Md.quantile(0.25)

for i in range(len(know)):
    point = know.drop(columns=["class"]).iloc[i]
    
    pBd = point - Bm
    pBd = pBd**2
    pBd = pBd.sum()
    pBd = np.sqrt(pBd)
    
    pMd = point - Mm
    pMd = pMd**2
    pMd = pMd.sum()
    pMd = np.sqrt(pMd)
    
    if pBd >= Bdm - Bdiqr and pBd <= Bdm + Bdiqr:
        true.append("normal")
    elif pMd >= Mdm - Mdiqr and pMd <= Mdm + Mdiqr:
        true.append("normal")
    else:
        true.append("anomaly")



# Glass

# one = test[test["class"]==1].drop(columns=["class"])
# onem = one.quantile(0.5)
# oned = one - onem
# oned = oned**2
# oned = oned.sum(axis=1)
# oned = np.sqrt(oned)
# onedm = oned.quantile(0.5)
# onediqr = oned.quantile(0.75)-oned.quantile(0.25)

# two = test[test["class"]==2].drop(columns=["class"])
# twom = two.quantile(0.5)
# twod = two - twom
# twod = twod**2
# twod = twod.sum(axis=1)
# twod = np.sqrt(twod)
# twodm = twod.quantile(0.5)
# twodiqr = twod.quantile(0.75)-twod.quantile(0.25)

# thr = test[test["class"]==3].drop(columns=["class"])
# thrm = thr.quantile(0.5)
# thrd = thr - thrm
# thrd = thrd**2
# thrd = thrd.sum(axis=1)
# thrd = np.sqrt(thrd)
# thrdm = thrd.quantile(0.5)
# thrdiqr = thrd.quantile(0.75)-thrd.quantile(0.25)

# fiv = test[test["class"]==5].drop(columns=["class"])
# fivm = fiv.quantile(0.5)
# fivd = fiv - fivm
# fivd = fivd**2
# fivd = fivd.sum(axis=1)
# fivd = np.sqrt(fivd)
# fivdm = fivd.quantile(0.5)
# fivdiqr = fivd.quantile(0.75)-fivd.quantile(0.25)

# six = test[test["class"]==6].drop(columns=["class"])
# sixm = six.quantile(0.5)
# sixd = six - sixm
# sixd = sixd**2
# sixd = sixd.sum(axis=1)
# sixd = np.sqrt(sixd)
# sixdm = sixd.quantile(0.5)
# sixdiqr = sixd.quantile(0.75)-sixd.quantile(0.25)

# sev = test[test["class"]==7].drop(columns=["class"])
# sevm = sev.quantile(0.5)
# sevd = sev - sevm
# sevd = sevd**2
# sevd = sevd.sum(axis=1)
# sevd = np.sqrt(sevd)
# sevdm = sevd.quantile(0.5)
# sevdiqr = sevd.quantile(0.75)-sevd.quantile(0.25)

# for i in range(len(know)):
#     point = know.drop(columns=["class"]).iloc[i]
    
#     poned = point - onem
#     poned = poned**2
#     poned = poned.sum()
#     poned = np.sqrt(poned)
    
#     ptwod = point - twom
#     ptwod = ptwod**2
#     ptwod = ptwod.sum()
#     ptwod = np.sqrt(ptwod)
    
#     pthrd = point - thrm
#     pthrd = pthrd**2
#     pthrd = pthrd.sum()
#     pthrd = np.sqrt(pthrd)
    
#     pfivd = point - fivm
#     pfivd = pfivd**2
#     pfivd = pfivd.sum()
#     pfivd = np.sqrt(pfivd)
    
#     psixd = point - sixm
#     psixd = psixd**2
#     psixd = psixd.sum()
#     psixd = np.sqrt(psixd)
    
#     psevd = point - sevm
#     psevd = psevd**2
#     psevd = psevd.sum()
#     psevd = np.sqrt(psevd)
    
#     if poned >= onedm - onediqr and poned <= onedm + onediqr:
#         true.append("normal")
#     elif ptwod >= twodm - twodiqr and ptwod <= twodm + twodiqr:
#         true.append("normal")
#     elif pthrd >= thrdm - thrdiqr and pthrd <= thrdm + thrdiqr:
#         true.append("normal")
#     elif pfivd >= fivdm - fivdiqr and pfivd <= fivdm + fivdiqr:
#         true.append("normal")
#     elif psixd >= sixdm - sixdiqr and psixd <= sixdm + sixdiqr:
#         true.append("normal")
#     elif psevd >= sevdm - sevdiqr and psevd <= sevdm + sevdiqr:
#         true.append("normal")
#     else:
#         true.append("anomaly")


# Parkinsons

# one = test[test["class"]==1].drop(columns=["class"])
# onem = one.quantile(0.5)
# oned = one - onem
# oned = oned**2
# oned = oned.sum(axis=1)
# oned = np.sqrt(oned)
# onedm = oned.quantile(0.5)
# onediqr = oned.quantile(0.75)-oned.quantile(0.25)

# zero = test[test["class"]==0].drop(columns=["class"])
# zerom = zero.quantile(0.5)
# zerod = zero - zerom
# zerod = zerod**2
# zerod = zerod.sum(axis=1)
# zerod = np.sqrt(zerod)
# zerodm = zerod.quantile(0.5)
# zerodiqr = zerod.quantile(0.75)-zerod.quantile(0.25)

# for i in range(len(know)):
#     point = know.drop(columns=["class"]).iloc[i]
    
#     poned = point - onem
#     poned = poned**2
#     poned = poned.sum()
#     poned = np.sqrt(poned)
    
#     pzerod = point - zerom
#     pzerod = pzerod**2
#     pzerod = pzerod.sum()
#     pzerod = np.sqrt(pzerod)

#     if poned >= onedm - onediqr and poned <= onedm + onediqr:
#         true.append("normal")
#     elif pzerod >= zerodm - zerodiqr and pzerod <= zerodm + zerodiqr:
#         true.append("normal")
#     else:
#         true.append("anomaly")

know["class"] = true

know = know[know["class"] == "anomaly"]


######################################
#### Exhaustive Parameter Testing ####
######################################


saved = [0,0,0,1,0,0,0,0,0,0]
best = [0,0,0,1,0,0,0,0,0,0]
#0 is True Anomaly
#1 is True Normal
#2 is False Anomaly
#3 is False Normal
#4 is k
#5 is tree
#6 is leaf
#7 is split
#8 is y
#9 is percent

progress = 1
k_test = [3, 5, 7]
tree_test = [5, 7, 9]
leaf_test = [5, 7, 9]
split_test = [5, 7, 9]
y_test = [10, 12, 14]
percent_test = [.3, .25, .2, .15]
outof = len(k_test)*len(tree_test)*len(leaf_test)*len(split_test)*len(y_test)*len(percent_test)

for k in k_test: 
    for tree in tree_test: 
        for leaf in leaf_test: 
            for split in split_test: 
                for y in y_test: 
                    for percent in percent_test: 


                        #### Generate Entries for Testing ####

                        field = planter(normal, tree, leaf, split)
                        
                        # More Knowledge
                        field = planter(testless, tree, leaf, split)
                        knowledge = planter(know.drop(columns = ["class"]), tree, leaf, split)
                        
                        # y = 10
                        acc = 0
                        
                        ilist = []
                        olist = []
                        
                        import random
                        
                        sample = []
                        for i in range(y*10):
                            # ran = random.randrange(0,len(normal))
                            # flowers = finder(normal.iloc[ran], field)
                            # c = closest_with_distance(normal.iloc[ran],flowers,k)["remove"]
                            
                            # More Knowledge
                            ran = random.randrange(0,len(testless))
                            flowers = finder(testless.iloc[ran], field)
                            c = closest_with_distance(testless.iloc[ran],flowers,k)["remove"]
                            
                            sample.append(c.mean())
                            
                        sample = np.sort(np.array(sample))
                        # percent = .2
                        last = int(percent*y*10) + 1
                        threshhold = sample[-last:].mean()
                        
                        anom = test.copy(deep=True)
                        
                        points = pd.DataFrame(columns = anom.columns)
                        
                        
                        
                        sum = 0
                        for i in range(y):
                            anomaly = non_anom_inner(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                            
                            
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(not pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Inner Points Within The Cluster Far From The Mean")
                        for i in range(y):
                            anomaly = non_anom_farin(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                            
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(not pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Inner Points Far From The Cluster")
                        for i in range(y):
                            anomaly = anom_farin(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                            
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Inner Points Close To The Cluster")
                        for i in range(y):
                            anomaly = anom_inner(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                                
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Outer Points Within The Cluster Close To The Mean")
                        for i in range(y):
                            anomaly = non_anom_outer(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                                
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(not pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Outer Points Within The Cluster Far From The Mean")
                        for i in range(y):
                            anomaly = non_anom_farout(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                                
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(not pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Outer Points Close To The Cluster")
                        for i in range(y):
                            anomaly = anom_outer(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                                
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        # print("")
                        
                        sum = 0
                        # print("Outer Point Far From The Cluster")
                        for i in range(y):
                            anomaly = anom_farout(std1, mean1, mean2)
                            anorm = (anomaly - test.drop(columns=["class"]).mean())/test.drop(columns=["class"]).std()
                            # flowers = finder(anorm, field)
                            # c = closest_with_distance(anorm,flowers,k)["remove"]
                            
                            
                            # More Knowledge
                            flowers = finder(anomaly, field)
                            c = closest_with_distance(anomaly,flowers,k)["remove"]
                            flowers = finder(anomaly, knowledge)
                            d = closest_with_distance(anomaly,flowers,k)["remove"]
                            
                            
                            # print("")
                            # print("Point: ",i)
                            # print("Average Distance: ",c.mean())
                            pred = anomaly_detector(c, threshhold)
                            if(pred):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"
                                
                            # More Knowledge
                            if (c.mean() > d.mean()):
                                anomaly["class"] = "anomaly"
                            else:
                                anomaly["class"] = "normal"    
                                
                                
                            anom.loc[len(anom.index)] = anomaly
                            points.loc[len(points.index)] = anomaly
                            anomaly.drop("class", inplace = True)
                            if(pred):
                                acc +=1
                            # print("Anomaly: ",pred)
                            #print(anomaly)
                            sum += c.mean()
                        # print("")
                        # print("Average Distance: ",sum/y)
                        
                        acc = acc/(y*8)
                        # print("")
                        # print("Accuracy = ",acc)
                        
                        
                        #### Determine True Values ####
                        
                        true = []
                        
                        # Iris
                        
                        # setosa = test[test["class"]=='Iris-setosa'].drop(columns=["class"])
                        # sm = setosa.quantile(0.5)
                        # sd = setosa - sm
                        # sd = sd**2
                        # sd = sd.sum(axis=1)
                        # sd = np.sqrt(sd)
                        # sdm = sd.quantile(0.5)
                        # sdiqr = sd.quantile(0.75)-sd.quantile(0.25)
                        
                        # versi = test[test["class"]=='Iris-versicolor'].drop(columns=["class"])
                        # vem = versi.quantile(0.5)
                        # ved = versi - vem
                        # ved = ved**2
                        # ved = ved.sum(axis=1)
                        # ved = np.sqrt(ved)
                        # vedm = ved.quantile(0.5)
                        # vediqr = ved.quantile(0.75)-ved.quantile(0.25)
                        
                        # virgin = test[test["class"]=='Iris-virginica'].drop(columns=["class"])
                        # vim = virgin.quantile(0.5)
                        # vid = virgin - vim
                        # vid = vid**2
                        # vid = vid.sum(axis=1)
                        # vid = np.sqrt(vid)
                        # vidm = vid.quantile(0.5)
                        # vidiqr = vid.quantile(0.75)-vid.quantile(0.25)
                        
                        # for i in range(len(points)):
                        #     point = points.drop(columns=["class"]).iloc[i]
                        #     psd = point - sm
                        #     psd = psd**2
                        #     psd = psd.sum()
                        #     psd = np.sqrt(psd)
                            
                        #     pved = point - vem
                        #     pved = pved**2
                        #     pved = pved.sum()
                        #     pved = np.sqrt(pved)
                            
                        #     pvid = point - vim
                        #     pvid = pvid**2
                        #     pvid = pvid.sum()
                        #     pvid = np.sqrt(pvid)
                            
                        #     if psd >= sdm - sdiqr and psd <= sdm + sdiqr:
                        #         true.append("normal")
                        #     elif pved >= vedm - vediqr and pved <= vedm + vediqr:
                        #         true.append("normal")
                        #     elif pvid >= vidm - vidiqr and pvid <= vidm + vidiqr:
                        #         true.append("normal")
                        #     else:
                        #         true.append("anomaly")
                        
                        
                        # Breast Cancer
                        
                        B = test[test["class"]=='B'].drop(columns=["class"])
                        Bm = B.quantile(0.5)
                        Bd = B - Bm
                        Bd = Bd**2
                        Bd = Bd.sum(axis=1)
                        Bd = np.sqrt(Bd)
                        Bdm = Bd.quantile(0.5)
                        Bdiqr = Bd.quantile(0.75)-Bd.quantile(0.25)
                        
                        M = test[test["class"]=='M'].drop(columns=["class"])
                        Mm = M.quantile(0.5)
                        Md = M - Mm
                        Md = Md**2
                        Md = Md.sum(axis=1)
                        Md = np.sqrt(Md)
                        Mdm = Md.quantile(0.5)
                        Mdiqr = Md.quantile(0.75)-Md.quantile(0.25)
                        
                        for i in range(len(points)):
                            point = points.drop(columns=["class"]).iloc[i]
                            
                            pBd = point - Bm
                            pBd = pBd**2
                            pBd = pBd.sum()
                            pBd = np.sqrt(pBd)
                            
                            pMd = point - Mm
                            pMd = pMd**2
                            pMd = pMd.sum()
                            pMd = np.sqrt(pMd)
                            
                            if pBd >= Bdm - Bdiqr and pBd <= Bdm + Bdiqr:
                                true.append("normal")
                            elif pMd >= Mdm - Mdiqr and pMd <= Mdm + Mdiqr:
                                true.append("normal")
                            else:
                                true.append("anomaly")
                        
                        
                        # Glass
                        
                        # one = test[test["class"]==1].drop(columns=["class"])
                        # onem = one.quantile(0.5)
                        # oned = one - onem
                        # oned = oned**2
                        # oned = oned.sum(axis=1)
                        # oned = np.sqrt(oned)
                        # onedm = oned.quantile(0.5)
                        # onediqr = oned.quantile(0.75)-oned.quantile(0.25)
                        
                        # two = test[test["class"]==2].drop(columns=["class"])
                        # twom = two.quantile(0.5)
                        # twod = two - twom
                        # twod = twod**2
                        # twod = twod.sum(axis=1)
                        # twod = np.sqrt(twod)
                        # twodm = twod.quantile(0.5)
                        # twodiqr = twod.quantile(0.75)-twod.quantile(0.25)
                        
                        # thr = test[test["class"]==3].drop(columns=["class"])
                        # thrm = thr.quantile(0.5)
                        # thrd = thr - thrm
                        # thrd = thrd**2
                        # thrd = thrd.sum(axis=1)
                        # thrd = np.sqrt(thrd)
                        # thrdm = thrd.quantile(0.5)
                        # thrdiqr = thrd.quantile(0.75)-thrd.quantile(0.25)
                        
                        # fiv = test[test["class"]==5].drop(columns=["class"])
                        # fivm = fiv.quantile(0.5)
                        # fivd = fiv - fivm
                        # fivd = fivd**2
                        # fivd = fivd.sum(axis=1)
                        # fivd = np.sqrt(fivd)
                        # fivdm = fivd.quantile(0.5)
                        # fivdiqr = fivd.quantile(0.75)-fivd.quantile(0.25)
                        
                        # six = test[test["class"]==6].drop(columns=["class"])
                        # sixm = six.quantile(0.5)
                        # sixd = six - sixm
                        # sixd = sixd**2
                        # sixd = sixd.sum(axis=1)
                        # sixd = np.sqrt(sixd)
                        # sixdm = sixd.quantile(0.5)
                        # sixdiqr = sixd.quantile(0.75)-sixd.quantile(0.25)
                        
                        # sev = test[test["class"]==7].drop(columns=["class"])
                        # sevm = sev.quantile(0.5)
                        # sevd = sev - sevm
                        # sevd = sevd**2
                        # sevd = sevd.sum(axis=1)
                        # sevd = np.sqrt(sevd)
                        # sevdm = sevd.quantile(0.5)
                        # sevdiqr = sevd.quantile(0.75)-sevd.quantile(0.25)
                        
                        # for i in range(len(points)):
                        #     point = points.drop(columns=["class"]).iloc[i]
                            
                        #     poned = point - onem
                        #     poned = poned**2
                        #     poned = poned.sum()
                        #     poned = np.sqrt(poned)
                            
                        #     ptwod = point - twom
                        #     ptwod = ptwod**2
                        #     ptwod = ptwod.sum()
                        #     ptwod = np.sqrt(ptwod)
                            
                        #     pthrd = point - thrm
                        #     pthrd = pthrd**2
                        #     pthrd = pthrd.sum()
                        #     pthrd = np.sqrt(pthrd)
                            
                        #     pfivd = point - fivm
                        #     pfivd = pfivd**2
                        #     pfivd = pfivd.sum()
                        #     pfivd = np.sqrt(pfivd)
                            
                        #     psixd = point - sixm
                        #     psixd = psixd**2
                        #     psixd = psixd.sum()
                        #     psixd = np.sqrt(psixd)
                            
                        #     psevd = point - sevm
                        #     psevd = psevd**2
                        #     psevd = psevd.sum()
                        #     psevd = np.sqrt(psevd)
                            
                        #     if poned >= onedm - onediqr and poned <= onedm + onediqr:
                        #         true.append("normal")
                        #     elif ptwod >= twodm - twodiqr and ptwod <= twodm + twodiqr:
                        #         true.append("normal")
                        #     elif pthrd >= thrdm - thrdiqr and pthrd <= thrdm + thrdiqr:
                        #         true.append("normal")
                        #     elif pfivd >= fivdm - fivdiqr and pfivd <= fivdm + fivdiqr:
                        #         true.append("normal")
                        #     elif psixd >= sixdm - sixdiqr and psixd <= sixdm + sixdiqr:
                        #         true.append("normal")
                        #     elif psevd >= sevdm - sevdiqr and psevd <= sevdm + sevdiqr:
                        #         true.append("normal")
                        #     else:
                        #         true.append("anomaly")
                        
                        
                        # Lung Cancer
                        
                        # one = test[test["class"]==1].drop(columns=["class"])
                        # onem = one.quantile(0.5)
                        # oned = one - onem
                        # oned = oned**2
                        # oned = oned.sum(axis=1)
                        # oned = np.sqrt(oned)
                        # onedm = oned.quantile(0.5)
                        # onediqr = oned.quantile(0.75)-oned.quantile(0.25)
                        
                        # two = test[test["class"]==2].drop(columns=["class"])
                        # twom = two.quantile(0.5)
                        # twod = two - twom
                        # twod = twod**2
                        # twod = twod.sum(axis=1)
                        # twod = np.sqrt(twod)
                        # twodm = twod.quantile(0.5)
                        # twodiqr = twod.quantile(0.75)-twod.quantile(0.25)
                        
                        # thr = test[test["class"]==3].drop(columns=["class"])
                        # thrm = thr.quantile(0.5)
                        # thrd = thr - thrm
                        # thrd = thrd**2
                        # thrd = thrd.sum(axis=1)
                        # thrd = np.sqrt(thrd)
                        # thrdm = thrd.quantile(0.5)
                        # thrdiqr = thrd.quantile(0.75)-thrd.quantile(0.25)
                        
                        # for i in range(len(points)):
                        #     point = points.drop(columns=["class"]).iloc[i]
                            
                        #     poned = point - onem
                        #     poned = poned**2
                        #     poned = poned.sum()
                        #     poned = np.sqrt(poned)
                            
                        #     ptwod = point - twom
                        #     ptwod = ptwod**2
                        #     ptwod = ptwod.sum()
                        #     ptwod = np.sqrt(ptwod)
                            
                        #     pthrd = point - thrm
                        #     pthrd = pthrd**2
                        #     pthrd = pthrd.sum()
                        #     pthrd = np.sqrt(pthrd)
                        
                        #     if poned >= onedm - onediqr and poned <= onedm + onediqr:
                        #         true.append("normal")
                        #     elif ptwod >= twodm - twodiqr and ptwod <= twodm + twodiqr:
                        #         true.append("normal")
                        #     elif pthrd >= thrdm - thrdiqr and pthrd <= thrdm + thrdiqr:
                        #         true.append("normal")
                        #     else:
                        #         true.append("anomaly")
                        
                        
                        # Parkinsons
                        
                        # one = test[test["class"]==1].drop(columns=["class"])
                        # onem = one.quantile(0.5)
                        # oned = one - onem
                        # oned = oned**2
                        # oned = oned.sum(axis=1)
                        # oned = np.sqrt(oned)
                        # onedm = oned.quantile(0.5)
                        # onediqr = oned.quantile(0.75)-oned.quantile(0.25)
                        
                        # zero = test[test["class"]==0].drop(columns=["class"])
                        # zerom = zero.quantile(0.5)
                        # zerod = zero - zerom
                        # zerod = zerod**2
                        # zerod = zerod.sum(axis=1)
                        # zerod = np.sqrt(zerod)
                        # zerodm = zerod.quantile(0.5)
                        # zerodiqr = zerod.quantile(0.75)-zerod.quantile(0.25)
                        
                        # for i in range(len(points)):
                        #     point = points.drop(columns=["class"]).iloc[i]
                            
                        #     poned = point - onem
                        #     poned = poned**2
                        #     poned = poned.sum()
                        #     poned = np.sqrt(poned)
                            
                        #     pzerod = point - zerom
                        #     pzerod = pzerod**2
                        #     pzerod = pzerod.sum()
                        #     pzerod = np.sqrt(pzerod)
                        
                        #     if poned >= onedm - onediqr and poned <= onedm + onediqr:
                        #         true.append("normal")
                        #     elif pzerod >= zerodm - zerodiqr and pzerod <= zerodm + zerodiqr:
                        #         true.append("normal")
                        #     else:
                        #         true.append("anomaly")
                        
                        
                        # Yeast
                        
                        # zero = test[test["class"]=="POX"].drop(columns=["class"])
                        # zerom = zero.quantile(0.5)
                        # zerod = zero - zerom
                        # zerod = zerod**2
                        # zerod = zerod.sum(axis=1)
                        # zerod = np.sqrt(zerod)
                        # zerodm = zerod.quantile(0.5)
                        # zerodiqr = zerod.quantile(0.75)-zerod.quantile(0.25)
                        
                        # one = test[test["class"]=="CYT"].drop(columns=["class"])
                        # onem = one.quantile(0.5)
                        # oned = one - onem
                        # oned = oned**2
                        # oned = oned.sum(axis=1)
                        # oned = np.sqrt(oned)
                        # onedm = oned.quantile(0.5)
                        # onediqr = oned.quantile(0.75)-oned.quantile(0.25)
                        
                        # two = test[test["class"]=="NUC"].drop(columns=["class"])
                        # twom = two.quantile(0.5)
                        # twod = two - twom
                        # twod = twod**2
                        # twod = twod.sum(axis=1)
                        # twod = np.sqrt(twod)
                        # twodm = twod.quantile(0.5)
                        # twodiqr = twod.quantile(0.75)-twod.quantile(0.25)
                        
                        # thr = test[test["class"]=="MIT"].drop(columns=["class"])
                        # thrm = thr.quantile(0.5)
                        # thrd = thr - thrm
                        # thrd = thrd**2
                        # thrd = thrd.sum(axis=1)
                        # thrd = np.sqrt(thrd)
                        # thrdm = thrd.quantile(0.5)
                        # thrdiqr = thrd.quantile(0.75)-thrd.quantile(0.25)
                        
                        # fou = test[test["class"]=="VAC"].drop(columns=["class"])
                        # foum = fou.quantile(0.5)
                        # foud = fou - foum
                        # foud = foud**2
                        # foud = foud.sum(axis=1)
                        # foud = np.sqrt(foud)
                        # foudm = foud.quantile(0.5)
                        # foudiqr = foud.quantile(0.75)-foud.quantile(0.25)
                        
                        # fiv = test[test["class"]=="ME3"].drop(columns=["class"])
                        # fivm = fiv.quantile(0.5)
                        # fivd = fiv - fivm
                        # fivd = fivd**2
                        # fivd = fivd.sum(axis=1)
                        # fivd = np.sqrt(fivd)
                        # fivdm = fivd.quantile(0.5)
                        # fivdiqr = fivd.quantile(0.75)-fivd.quantile(0.25)
                        
                        # six = test[test["class"]=="ME2"].drop(columns=["class"])
                        # sixm = six.quantile(0.5)
                        # sixd = six - sixm
                        # sixd = sixd**2
                        # sixd = sixd.sum(axis=1)
                        # sixd = np.sqrt(sixd)
                        # sixdm = sixd.quantile(0.5)
                        # sixdiqr = sixd.quantile(0.75)-sixd.quantile(0.25)
                        
                        # sev = test[test["class"]=="ME1"].drop(columns=["class"])
                        # sevm = sev.quantile(0.5)
                        # sevd = sev - sevm
                        # sevd = sevd**2
                        # sevd = sevd.sum(axis=1)
                        # sevd = np.sqrt(sevd)
                        # sevdm = sevd.quantile(0.5)
                        # sevdiqr = sevd.quantile(0.75)-sevd.quantile(0.25)
                        
                        # eig = test[test["class"]=="EXC"].drop(columns=["class"])
                        # eigm = eig.quantile(0.5)
                        # eigd = eig - eigm
                        # eigd = eigd**2
                        # eigd = eigd.sum(axis=1)
                        # eigd = np.sqrt(eigd)
                        # eigdm = eigd.quantile(0.5)
                        # eigdiqr = eigd.quantile(0.75)-eigd.quantile(0.25)
                        
                        # nin = test[test["class"]=="ERL"].drop(columns=["class"])
                        # ninm = nin.quantile(0.5)
                        # nind = nin - ninm
                        # nind = nind**2
                        # nind = nind.sum(axis=1)
                        # nind = np.sqrt(nind)
                        # nindm = nind.quantile(0.5)
                        # nindiqr = nind.quantile(0.75)-nind.quantile(0.25)
                        
                        # for i in range(len(points)):
                        #     point = points.drop(columns=["class"]).iloc[i]
                            
                        #     pzerod = point - zerom
                        #     pzerod = pzerod**2
                        #     pzerod = pzerod.sum()
                        #     pzerod = np.sqrt(pzerod)
                            
                        #     poned = point - onem
                        #     poned = poned**2
                        #     poned = poned.sum()
                        #     poned = np.sqrt(poned)
                            
                        #     ptwod = point - twom
                        #     ptwod = ptwod**2
                        #     ptwod = ptwod.sum()
                        #     ptwod = np.sqrt(ptwod)
                            
                        #     pthrd = point - thrm
                        #     pthrd = pthrd**2
                        #     pthrd = pthrd.sum()
                        #     pthrd = np.sqrt(pthrd)
                            
                        #     pfoud = point - foum
                        #     pfoud = pfoud**2
                        #     pfoud = pfoud.sum()
                        #     pfoud = np.sqrt(pfoud)
                            
                        #     pfivd = point - fivm
                        #     pfivd = pfivd**2
                        #     pfivd = pfivd.sum()
                        #     pfivd = np.sqrt(pfivd)
                            
                        #     psixd = point - sixm
                        #     psixd = psixd**2
                        #     psixd = psixd.sum()
                        #     psixd = np.sqrt(psixd)
                            
                        #     psevd = point - sevm
                        #     psevd = psevd**2
                        #     psevd = psevd.sum()
                        #     psevd = np.sqrt(psevd)
                            
                        #     peigd = point - eigm
                        #     peig = peigd**2
                        #     peigd = peigd.sum()
                        #     peigd = np.sqrt(peigd)
                            
                        #     pnind = point - ninm
                        #     pnin = pnind**2
                        #     pnind = pnind.sum()
                        #     pnind = np.sqrt(pnind)
                            
                        #     if poned >= onedm - onediqr and poned <= onedm + onediqr:
                        #         true.append("normal")
                        #     elif ptwod >= twodm - twodiqr and ptwod <= twodm + twodiqr:
                        #         true.append("normal")
                        #     elif pthrd >= thrdm - thrdiqr and pthrd <= thrdm + thrdiqr:
                        #         true.append("normal")
                        #     elif pfoud >= foudm - foudiqr and pfoud <= foudm + foudiqr:
                        #         true.append("normal")
                        #     elif pfivd >= fivdm - fivdiqr and pfivd <= fivdm + fivdiqr:
                        #         true.append("normal")
                        #     elif psixd >= sixdm - sixdiqr and psixd <= sixdm + sixdiqr:
                        #         true.append("normal")
                        #     elif psevd >= sevdm - sevdiqr and psevd <= sevdm + sevdiqr:
                        #         true.append("normal")
                        #     elif peigd >= eigdm - eigdiqr and peigd <= eigdm + eigdiqr:
                        #         true.append("normal")
                        #     elif pnind >= nindm - nindiqr and pnind <= nindm + nindiqr:
                        #         true.append("normal")
                        #     elif pzerod >= zerodm - zerodiqr and pzerod <= zerodm + zerodiqr:
                        #         true.append("normal")
                        #     else:
                        #         true.append("anomaly")
                        
                        
                        #### Store Best Results ####
                        
                        points["actual"] = true
                        
                        # print("K = ",k)
                        # print("Tree = ",tree)
                        # print("Leaf = ",leaf)
                        # print("Split = ",split)
                        # print("Y = ",y)
                        # print("Percent = ",percent)
                        # print("")
                        # print("Trues")
                        # print(points[points["class"]==points["actual"]]["class"].value_counts())
                        # print("Falses")
                        # print(points[points["class"]!=points["actual"]]["class"].value_counts())
                        # print("")
                        
                        tnorm = points[points["class"]==points["actual"]]
                        tnorm = len(tnorm[tnorm["class"]=="normal"])
                        tanom = points[points["class"]==points["actual"]]
                        tanom = len(tanom[tanom["class"]=="anomaly"])
                        fnorm = points[points["class"]!=points["actual"]]
                        fnorm = len(fnorm[fnorm["class"]=="normal"])
                        fanom = points[points["class"]!=points["actual"]]
                        fanom = len(fanom[fanom["class"]=="anomaly"])
                        
                        if tanom/(tanom+fnorm) > saved[0]/(saved[0]+saved[3]):
                            saved = [tanom, tnorm, fanom, fnorm, k, tree, leaf, split, y, percent]
                        if (tanom+tnorm)/(tanom+fnorm+tnorm+fanom) > (best[0]+best[1])/(best[0]+best[1]+best[2]+best[3]):
                            best = [tanom, tnorm, fanom, fnorm, k, tree, leaf, split, y, percent]
                        print("Progress = ", int((progress/outof)*10000)/100,"%")
                        print("")
                        progress += 1
                        
                        
                        
                        
                        # Using Additional Knowledge
                        
                        
#### Report Best Outcome ####

print("True Anomaly Rate")
print("")

print("True Anomalies = ", saved[0])
print("True Normals = ", saved[1])
print("False Anomalies = ", saved[2])
print("False Normals = ", saved[3])
print("")

print("K = ",saved[4])
print("Tree = ",saved[5])
print("Leaf = ",saved[6])
print("Split = ",saved[7])
print("Y = ",saved[8])
print("Percent = ",saved[9])

print("")
print("")
print("")

print("Accuracy")
print("")

print("True Anomalies = ", best[0])
print("True Normals = ", best[1])
print("False Anomalies = ", best[2])
print("False Normals = ", best[3])
print("")

print("K = ",best[4])
print("Tree = ",best[5])
print("Leaf = ",best[6])
print("Split = ",best[7])
print("Y = ",best[8])
print("Percent = ",best[9])
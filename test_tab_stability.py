import sys
import os

PATH = os.getcwd()
sys.path.append(PATH)

import stability as st #Nogueira, Sechidis, Brown.

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict
import random
import joblib
import json

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC

import lime
import lime.lime_tabular
from lime import submodular_pick


from tqdm import tqdm

import shap

import warnings
warnings.filterwarnings('ignore')

from lime import submodular_pick

def generate_lime_explanations(explainer,test_xi, cls, submod=False, test_all_data=None, max_feat = 10, scaler=None):
    
    def scale_predict_fn(X):
        scaled_data = scaler.transform(X)
        pred = cls.predict_proba(scaled_data)
        return pred

    def predict_fn(X):
        #X = X.reshape(1, -1)
        pred = cls.predict_proba(X)
        return pred

    if scaler == None:
        exp = explainer.explain_instance(test_xi, 
                                 predict_fn, num_features=max_feat, labels=[0,1])
    else:
        exp = explainer.explain_instance(test_xi, 
                                 scale_predict_fn, num_features=max_feat, labels=[0,1])
        
    return exp
        
def dispersal(weights, features):
    
    feat_len = len(features)
    weights_by_feat = []
    
    #Weights are sorted by iteration. Transpose list.
    for i in list(range(feat_len)):
        feat_weight = []
        for iteration in weights:
            feat_weight.append(iteration[i])
        weights_by_feat.append(feat_weight)
    
    dispersal = []
    dispersal_no_outlier = []
    
    for each in weights_by_feat:
        #Find mean and variance of weight for each feature
        mean = np.mean(each)
        std_dev = np.std(each)
        var = std_dev**2
        
        #Calculate relative variance, ignore features where the weight is always 0
        if mean == 0:
            dispersal.append(0)
            dispersal_no_outlier.append(0)
        else:
            rel_var = var/abs(mean)
            dispersal.append(rel_var)
            
            #dispersal without outliers - remove anything with a z-score higher
            #than 3 (more than 3 standard deviations away from the mean)
            rem_outlier = []
            z_scores = stats.zscore(each)

            for i in range(len(z_scores)):
                if -3 < z_scores[i] < 3:
                    rem_outlier.append(each[i])
            if rem_outlier != []:
                new_mean = np.mean(rem_outlier)
                if new_mean == 0:
                    dispersal_no_outlier.append(0)
                else:
                    new_std = np.std(rem_outlier)
                    new_var = new_std**2
                    new_rel_var = new_var/abs(new_mean)
                    dispersal_no_outlier.append(new_rel_var)
            else:
                dispersal_no_outlier.append(rel_var)

    return dispersal, dispersal_no_outlier

def create_samples(shap_explainer, iterations, row, features, pred, top = None, scaler = None):
    length = len(features)
    
    exp = []
    rel_exp = []
    
    if scaler != None:
        row = scaler.transform(row)
    
    for j in range(iterations):
        #Generate shap values for row
        if type(shap_explainer) == shap.explainers._tree.Tree:
            shap_values = shap_explainer(row, check_additivity = False).values
        else:
            shap_values = shap_explainer(row.reshape(1, -1)).values
        
        #print(exp.shape)
        #print(exp)
        #print(shap_values.shape)
        #print(len(features))
        if shap_values.shape == (1, len(features), 2):
            shap_values = shap_values[0]
            
        #print(exp.shape)
        
        if shap_values.shape == (len(features), 2):
            shap_values = np.array([feat[pred] for feat in shap_values]).reshape(len(features))
        elif shap_values.shape == (1, len(row)) or shap_values.shape == (len(features), 1):
            shap_values = shap_values.reshape(len(features))
            
        #print(np.array(exp).shape)
        
        if scaler != None:
            #print(shap_values)
            shap_values = scaler.inverse_transform(shap_values.reshape(1, -1))[0]
            #print(shap_values.shape)
        
        #Map SHAP values to feature names
        importances = []
        
        abs_values = []
    
        for i in range(length):
            feat = features[i]
            shap_val = shap_values[i]
            abs_val = abs(shap_values[i])
            entry = (feat, shap_val, abs_val)
            importances.append(entry)
            abs_values.append(abs_val)
        
        #Sort features by influence on result
        importances.sort(key=lambda tup: tup[2], reverse = True)
        
        #Create list of all feature
        exp.append(importances)
        
        #print(exp[0])
        
        #Create list of most important features
        rel_feat = []
        if top != None:
            for i in range(top):
                feat = importances[i]
                if feat[2] > 0:
                    rel_feat.append(feat)

            rel_exp.append(rel_feat)
        else:
            bins = pd.cut(abs_values, 4, duplicates = "drop", retbins = True)[-1]
            q1_min = bins[-2]
            rel_feat = [feat for feat in importances if feat[2] > q1_min]
            rel_exp.append(rel_feat)
        
    return exp, rel_exp

dataset_ref = sys.argv[1]
cls_method = sys.argv[2]

xai_method = sys.argv[3]

sample_size = 2
exp_iter = 5
max_feat = 10

dataset_path = os.path.join(PATH, dataset_ref)

cls = joblib.load(os.path.join(dataset_path, cls_method, "cls.joblib"))
scaler = joblib.load(os.path.join(dataset_path, "scaler.joblib"))

trainingdata = pd.read_csv(os.path.join(dataset_path, "datasets", dataset_ref+"_Xtrain.csv"), sep=";")
with open(os.path.join(dataset_path, "datasets",'col_dict.json')) as file:
    col_dict = json.load(file)
file.close()

sample_instances = pd.read_csv((os.path.join(dataset_path, cls_method, "test_sample.csv")), sep=";") 
results = pd.read_csv((os.path.join(dataset_path, cls_method, "results.csv")), sep=";")

if xai_method=="SHAP":
    
    if cls_method == "xgboost" or cls_method == "decision_tree":
        shap_explainer = shap.Explainer(cls)
    elif cls_method == "nb":
        shap_explainer = shap.Explainer(cls.predict_proba, trainingdata)
    else:
        shap_explainer = shap.Explainer(cls, trainingdata)
    print(type(shap_explainer))

    feat_list = trainingdata.columns.tolist()

    subset_stability = []
    weight_stability = []
    adjusted_weight_stability = []

    #explain the chosen instances and find the stability score
    instance_no = 0
    for instance in tqdm(sample_instances.values):
        instance_no += 1    
        print("Testing", instance_no, "of", len(sample_instances), ".")

        #if cls_method == "xgboost":
        instance = instance.reshape(1, -1)
        pred = cls.predict(instance)

        #Get Tree SHAP explanations for instance
        exp, rel_exp = create_samples(shap_explainer, exp_iter, instance, feat_list, pred, scaler = scaler)

        feat_pres = []
        feat_weights = []

        for iteration in rel_exp:
            #print("Computing feature presence for iteration", rel_exp.index(iteration))

            presence_list = [0]*len(feat_list)

            for each in feat_list:
                list_idx = feat_list.index(each)

                for explanation in iteration:
                    if each in explanation[0]:
                        presence_list[list_idx] = 1

            feat_pres.append(presence_list)

        for iteration in exp:
            #print("Compiling feature weights for iteration", exp.index(iteration))

            weights = [0]*len(feat_list)

            for each in feat_list:
                list_idx = feat_list.index(each)

                for explanation in iteration:
                    if each in explanation[0]:

                        weights[list_idx] = explanation[1]
            feat_weights.append(weights)

        stability = st.getStability(feat_pres)
        print ("Stability:", round(stability,2))
        subset_stability.append(stability)

        rel_var, second_var = dispersal(feat_weights, feat_list)
        avg_dispersal = 1-np.mean(rel_var)
        print ("Dispersal of feature importance:", round(avg_dispersal, 2))
        weight_stability.append(avg_dispersal)
        adj_dispersal = 1-np.mean(second_var)
        print ("Dispersal with no outliers:", round(adj_dispersal, 2))
        adjusted_weight_stability.append(adj_dispersal)

    results["SHAP Subset Stability"] = subset_stability
    results["SHAP Weight Stability"] = weight_stability
    results["SHAP Adjusted Weight Stability"] = adjusted_weight_stability
    
if xai_method=="LIME":
        
    feat_list = trainingdata.columns.tolist()
    class_names = ["Negative", "Positive"]

    cats = col_dict["discrete"]

    subset_stability = []
    weight_stability = []
    adjusted_weight_stability = []

    #create explainer now that can be passed later
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata.values,
                          feature_names = feat_list, class_names=class_names, categorical_features = cats)

    instance_no = 0
    print(len(sample_instances))
    #explain the chosen instances and find the stability score
    for instance in tqdm(sample_instances.values):
        instance_no += 1

        print("Testing", instance_no, "of", len(sample_instances), ".")

        #Get lime explanations for instance
        feat_pres = []
        feat_weights = []

        for iteration in list(range(exp_iter)):

            lime_exp = generate_lime_explanations(lime_explainer, instance, cls,
                                                  max_feat = len(feat_list), scaler = scaler)

            all_weights = [exp[1] for exp in lime_exp.as_list()]
            bins = pd.cut(all_weights, 4, duplicates = "drop", retbins = True)[-1]
            q1_min = bins[-2]

            presence_list = [0]*len(feat_list)
            weights = [0]*len(feat_list)

            for each in feat_list:
                list_idx = feat_list.index(each)
                #print ("Feature", list_idx)
                for explanation in lime_exp.as_list():
                    if each in explanation[0]:
                        if explanation[1] > q1_min:
                            presence_list[list_idx] = 1
                        weights[list_idx] = explanation[1]

            feat_pres.append(presence_list)
            feat_weights.append(weights)

        stability = st.getStability(feat_pres)
        print ("Stability:", round(stability,2))
        subset_stability.append(stability)

        rel_var, second_var = dispersal(feat_weights, feat_list)
        avg_dispersal = 1-np.mean(rel_var)
        print ("Dispersal of feature importance:", round(avg_dispersal, 2))
        weight_stability.append(avg_dispersal)
        adj_dispersal = 1-np.mean(second_var)
        print ("Dispersal with no outliers:", round(adj_dispersal, 2))
        adjusted_weight_stability.append(adj_dispersal)

    results["LIME Subset Stability"] = subset_stability
    results["LIME Weight Stability"] = weight_stability
    results["LIME Adjusted Weight Stability"] = adjusted_weight_stability

results.to_csv(os.path.join(dataset_path, cls_method, "results.csv"))
print("Results saved")

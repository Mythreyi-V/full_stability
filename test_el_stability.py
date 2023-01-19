import sys
import os

#Use if working on Colab
#from google.colab import drive
#drive.mount('/content/drive')
#PATH = '/content/drive/My Drive/PPM_Stability/'

#If working locally
PATH = os.getcwd()
sys.path.append(PATH)

import EncoderFactory
#from DatasetManager_for_colab import DatasetManager
from DatasetManager import DatasetManager
import BucketFactory
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

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import lime
import lime.lime_tabular
from lime import submodular_pick

from anchor import anchor_tabular
#from alibi.utils.data import gen_category_map

import shap

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

def imp_df(column_names, importances):
        df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
        return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title, num_feat):
        imp_df.columns = ['feature', 'feature_importance']
        b= sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df.head(num_feat), orient = 'h', palette="Blues_r")

from lime import submodular_pick

def generate_lime_explanations(explainer,test_xi, cls, submod=False, test_all_data=None, max_feat = 10, scaler=None):
    
    #print("Actual value ", test_y)
    
   # print(type(test_xi))
   # print(type(cls.predict_proba))
   # print(type(max_feat))
    def scale_predict_fn(X):
        scaled_data = scaler.transform(X)
        pred = cls.predict_proba(scaled_data)
        return pred
            
    if scaler == None:
        exp = explainer.explain_instance(test_xi, 
                                 cls.predict_proba, num_features=max_feat, labels=[0,1])
    else:
        exp = explainer.explain_instance(test_xi, 
                                 scale_predict_fn, num_features=max_feat, labels=[0,1])
        
    return exp
        
    if submod==True:
        sp_obj=submodular_pick.SubmodularPick(explainer, test_all_data, cls.predict_proba, 
                                      sample_size=20, num_features=num_features,num_exps_desired=4)
        [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations];

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
            shap_values = scaler.inverse_transform(shap_values.reshape(1, -1))
            print(shap_values.shape)
        
        #Map SHAP values to feature names
        importances = []
        
        abs_values = []
    
        for i in range(length):
            feat = features[i]
            shap_val = shap_values[0][i]
            abs_val = abs(shap_values[0][i])
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
bucket_method = sys.argv[2]
cls_encoding = sys.argv[3]
cls_method = sys.argv[4]

gap = 1
n_iter = 1

method_name = "%s_%s"%(bucket_method, cls_encoding)

xai_method = sys.argv[5]

sample_size = 2
exp_iter = 10
max_feat = 10
max_prefix = 20

dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
    "bpic2017" : ["bpic2017_accepted"],
    "bpic2012" : ["bpic2012_accepted"],
    #"insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1"],# "sepsis_cases_2", "sepsis_cases_4"]
    "production": ["production"] 
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

for dataset_name in datasets:

    min_prefix_length = 1

    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    all_cls = []
    all_encoders = []
    all_scalers = []
    all_train = []
    all_samples = []
    all_results = []
    
    sample = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample.csv" % 
                                      (dataset_ref, cls_method, method_name)))
    results_template = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results.csv" % 
                                      (dataset_ref, cls_method, method_name)))
    
    for ii in range(n_iter):
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% (dataset_ref, cls_method, method_name)))])

        for bucket in range(num_buckets):
            bucketID = bucket+1
            print ('Bucket', bucketID)

            #import everything needed to sort and predict
            pipeline_path = os.path.join(PATH, "%s/%s/%s/pipelines/pipeline_bucket_%s.joblib" % 
                                         (dataset_ref, cls_method, method_name, bucketID))
            pipeline = joblib.load(pipeline_path)
            feature_combiner = pipeline['encoder']
            if 'scaler' in pipeline.named_steps:
                scaler = pipeline['scaler']
            else:
                scaler = None
            cls = pipeline['cls']
            
            all_cls.append(cls)
            all_encoders.append(feature_combiner)
            all_scalers.append(scaler)

            #find relevant samples for bucket
            if bucket_method == "prefix":
                bucket_sample = sample[sample["prefix_nr"] == bucket]
                bucket_results = results_template[results_template["Prefix Length"] == bucket]
            else:
                bucket_sample = sample
                bucket_results = results_template
            
            feat_names = feature_combiner.get_feature_names()
            feat_list = [feat.replace(" ", "_") for feat in feat_names]
            
            encoded_sample = pd.DataFrame(columns = feat_list)
            
            for _,group in bucket_sample.groupby(dataset_manager.case_id_col):
                row = feature_combiner.transform(group).reshape(-1)
                
                row_data = {}
                for col in range(len(encoded_sample.columns)):
                    row_data[encoded_sample.columns[col]] = row[col]
                
                encoded_sample = encoded_sample.append(row_data, ignore_index = True)
                
            all_samples.append(encoded_sample)
            all_results.append(bucket_results)
            
            #import training data for bucket
            dt_training_bucket = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID)))
            train_data = feature_combiner.transform(dt_training_bucket)
            if scaler != None:
                train_data = scaler.fit_transform(train_data)
            
            all_train.append(train_data)

if xai_method=="SHAP":

    for dataset_name in datasets:
        
        for ii in range(n_iter):
            num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                        (dataset_ref, cls_method, method_name)))])
            
            for bucket in range(num_buckets):
                bucketID = bucket+1
                print ('Bucket', bucketID)

                cls = all_cls[bucket]
                feature_combiner = all_encoders[bucket]
                scaler = all_scalers[bucket]
                trainingdata = all_train[bucket]
                sample_instances = all_samples[bucket]
                results = all_results[bucket]
                
                if cls_method == "xgboost" or cls_method == "decision_tree":
                    shap_explainer = shap.Explainer(cls)
                elif cls_method == "nb":
                    shap_explainer = shap.Explainer(cls.predict_proba, trainingdata)
                else:
                    shap_explainer = shap.Explainer(cls, trainingdata)
                print(type(shap_explainer))
                
                feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
                
                subset_stability = []
                weight_stability = []
                adjusted_weight_stability = []
                    
                #explain the chosen instances and find the stability score
                instance_no = 0
                for instance in sample_instances[:10]:
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
                all_results[bucket] = results
        
if xai_method=="LIME":

    for dataset_name in datasets:
        
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                    (dataset_ref, cls_method, method_name)))])
        dataset_manager = DatasetManager(dataset_name)

        for bucket in range(num_buckets):
            bucketID = bucket+1
            print ('Bucket', bucketID)
            
            cls = all_cls[bucket]
            feature_combiner = all_encoders[bucket]
            scaler = all_scalers[bucket]
            trainingdata = all_train[bucket]
            sample_instances = all_samples[bucket]
            results = all_results[bucket]
            pipeline = all_pipelines[bucket]            

            feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
            class_names = ["Negative", "Positive"]
            
            cats = [feat for col in dataset_manager.dynamic_cat_cols+dataset_manager.static_cat_cols 
                    for feat in range(len(feat_list)) if col in feat_list[feat]]

            subset_stability = []
            weight_stability = []
            adjusted_weight_stability = []

            #create explainer now that can be passed later
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
                                  feature_names = feat_list, class_names=class_names, categorical_features = cats)
            
            instance_no = 0
            print(len(sample_instances))
            #explain the chosen instances and find the stability score
            for instance in sample_instances[:10]:
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
            all_results[bucket] = results

pd.concat(all_results).to_csv(os.path.join(PATH,"%s/%s/%s/samples/results.csv") % (dataset_ref, cls_method, method_name))

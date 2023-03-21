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

#from alibi.utils.data import gen_category_map

from tqdm import tqdm_notebook, tqdm

from acv_explainers import ACXplainer
from learning import *
import pyAgrum

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
            rel_feat = [feat for feat in importances if feat[2] >= q1_min]
            rel_exp.append(rel_feat)
        
    return exp, rel_exp

def get_acv_features(explainer, instance, cls, X_train, y_train, exp_iter):
    instance = instance.reshape(1, -1)
    y = cls.predict(instance)
    
    t=np.var(y_train)

    feats = []
    feat_imp = []

    for i in range(exp_iter):
        sufficient_expl, sdp_expl, sdp_global = explainer.sufficient_expl_rf(instance, y, X_train, y_train,
                                                                                 t=t, pi_level=0.8)
        clean_expl = sufficient_expl.copy()
        clean_expl = clean_expl[0]
        clean_expl = [sublist for sublist in clean_expl if sum(n<0 for n in sublist)==0 ]

        clean_sdp = sdp_expl[0].copy()
        clean_sdp = [sdp for sdp in clean_sdp if sdp > 0]
        
        lximp = explainer.compute_local_sdp(X_train.shape[1], clean_expl)
        feat_imp.append(lximp)
        
        if len(clean_expl)==0 or len(clean_expl[0])==0:
            print("No explamation meets pi level")
        else:
            lens = [len(i) for i in clean_expl]
            me_loc = [i for i in range(len(lens)) if lens[i]==min(lens)]
            mse_loc = np.argmax(np.array(clean_sdp)[me_loc])
            mse = np.array(clean_expl)[me_loc][mse_loc]
            feats.extend(mse)

    if len(feats)==0:
        feat_pos = []
    else:
        feat_pos = set(feats)
    
      
    feat_imp = np.mean(feat_imp, axis=0)
    
    return feat_imp, feat_pos

def get_linda_features(instance, cls, scaler, dataset, exp_iter, feat_list, percentile):
    label_lst = ["Negative", "Positive"]
    
    feat_pos = []
    lkhoods = []
    
    save_to = os.path.join(PATH, dataset, cls_method, method_name)+"/"
    
    for i in range(exp_iter):
        [bn, inference, infoBN] = generate_BN_explanations(instance, label_lst, feat_list, "Result", 
                                                           None, scaler, cls, save_to, dataset, show_in_notebook = False,
                                                           samples=round(len(feat_list)*1.5))
        
        ie = pyAgrum.LazyPropagation(bn)
        result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
        result_proba = result_posterior.loc["Result", label_lst[instance['predictions']]]
        row = instance['original_vector']
        #print(row)

        likelihood = [0]*len(feat_list)

        for j in range(len(feat_list)):
            var_labels = bn.variable(feat_list[j]).labels()
            str_bins = list(var_labels)
            bins = []

            for disc_bin in str_bins:
                disc_bin = disc_bin.strip('"(]')
                cat = [float(val) for val in disc_bin.split(',')]
                bins.append(cat)

            for k in range(len(bins)):
                if k == 0 and row[j] <= bins[k][0]:
                    feat_bin = str_bins[k]
                elif k == len(bins)-1 and row[j] >= bins[k][1]:
                    feat_bin = str_bins[k]
                elif row[j] > bins[k][0] and row[j] <= bins[k][1]:
                    feat_bin = str_bins[k]

            ie = pyAgrum.LazyPropagation(bn)
            ie.setEvidence({feat_list[j]: feat_bin})
            ie.makeInference()
            
            result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
            new_proba = result_posterior.loc["Result", label_lst[instance['predictions']]]
            #print(result_proba, new_proba)
            proba_change = result_proba-new_proba
            likelihood[j] = abs(proba_change)

        lkhoods.append(likelihood)
        
    min_coef = min( np.mean(lkhoods, axis=0))
    max_coef = max( np.mean(lkhoods, axis=0))
    
    k = (max_coef-min_coef)*percentile
    q1_min = max_coef - k

    #If fixing all features produces the same result for the class,
    #return all features
    if len(set(np.mean(lkhoods, axis=0)))==1:
        feat_pos.extend(range(len(feat_list)))
    else:
        feat_pos.extend(list(np.where(np.mean(lkhoods, axis=0) >= q1_min)[0]))

    feat_pos = set(feat_pos)
    
    return np.mean(lkhoods, axis=0), feat_pos

dataset_ref = sys.argv[1]
params_dir = PATH + "params"
results_dir = "results"
bucket_method = sys.argv[2]
cls_encoding = sys.argv[3]
cls_method = sys.argv[4]

gap = 1
n_iter = 1

method_name = "%s_%s"%(bucket_method, cls_encoding)
save_to = os.path.join(PATH, dataset_ref, cls_method, method_name)

xai_method = sys.argv[5]

sample_size = 2
exp_iter = 5
max_feat = 10
max_prefix = 20
random_state = 22

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

all_results = []

if xai_method=="SHAP":

    for dataset_name in datasets:
        
        for ii in range(n_iter):
            num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                        (dataset_ref, cls_method, method_name)))])
            
            for bucket in tqdm_notebook(range(num_buckets)):
                bucketID = bucket+1
                print ('Bucket', bucketID)
                
                #import everything needed to sort and predict
                pipeline_path = os.path.join(PATH, save_to, "pipelines/pipeline_bucket_%s.joblib" % 
                                             (bucketID))
                pipeline = joblib.load(pipeline_path)
                feature_combiner = pipeline['encoder']
                if 'scaler' in pipeline.named_steps:
                    scaler = pipeline['scaler']
                else:
                    scaler = None
                cls = pipeline['cls']
                
                #import training data for bucket
                trainingdata = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                              (dataset_ref, cls_method, method_name, bucketID))).values
                targets = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/y_train_bucket_%s.csv" % 
                                                              (dataset_ref, cls_method, method_name, bucketID))).values
                if scaler != None:
                    trainingdata = scaler.transform(trainingdata)
                    
                #find relevant samples for bucket
                sample_instances = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample_bucket_%s.csv" % 
                                          (dataset_ref, cls_method, method_name, bucketID))).values
                results = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results_bucket_%s.csv" % 
                                          (dataset_ref, cls_method, method_name, bucketID)), sep=";")
                
                if scaler != None:
                    sample_instances = scaler.transform(sample_instances)
                
                #create explanation mechanism
                if cls_method == "xgboost" or cls_method == "decision_tree":
                    shap_explainer = shap.Explainer(cls)
                elif cls_method == "nb":
                    shap_explainer = shap.Explainer(cls.predict_proba, trainingdata)
                else:
                    shap_explainer = shap.Explainer(cls, trainingdata)
                print(type(shap_explainer))
                
                #Identify feature names
                feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
                
                subset_stability = []
                weight_stability = []
                adjusted_weight_stability = []
                    
                #explain the chosen instances and find the stability score
                instance_no = 0
                for instance in tqdm_notebook(sample_instances):
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
                results.to_csv(os.path.join(PATH,"%s/%s/%s/samples/results_bucket_%s.csv") % 
                               (dataset_ref, cls_method, method_name, bucketID), sep=";", index=False)
                
                all_results.append(results)

if xai_method=="LIME":

    for dataset_name in datasets:
        
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                    (dataset_ref, cls_method, method_name)))])
        dataset_manager = DatasetManager(dataset_name)

        for bucket in tqdm_notebook(range(num_buckets)):
            bucketID = bucket+1
            print ('Bucket', bucketID)
            
           #import everything needed to sort and predict
            pipeline_path = os.path.join(PATH, save_to, "pipelines/pipeline_bucket_%s.joblib" % 
                                         (bucketID))
            pipeline = joblib.load(pipeline_path)
            feature_combiner = pipeline['encoder']
            if 'scaler' in pipeline.named_steps:
                scaler = pipeline['scaler']
            else:
                scaler = None
            cls = pipeline['cls']

            #import training data for bucket
            trainingdata = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            targets = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/y_train_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            if scaler != None:
                trainingdata = scaler.transform(trainingdata)

            #find relevant samples for bucket
            sample_instances = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID))).values
            results = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID)), sep=";")
            
            if scaler != None:
                sample_instances = scaler.transform(sample_instances)

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
            for instance in tqdm_notebook(sample_instances):
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
                                if explanation[1] >= q1_min:
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
            results.to_csv(os.path.join(PATH,"%s/%s/%s/samples/results_bucket_%s.csv") % 
                               (dataset_ref, cls_method, method_name, bucketID), sep=";", index=False)
                
            all_results.append(results)

if xai_method=="ACV":

    for dataset_name in datasets:
        
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                    (dataset_ref, cls_method, method_name)))])
        dataset_manager = DatasetManager(dataset_name)

        for bucket in tqdm_notebook(range(num_buckets)):
            bucketID = bucket+1
            print ('Bucket', bucketID)
            
            #import everything needed to sort and predict
            pipeline_path = os.path.join(PATH, save_to, "pipelines/pipeline_bucket_%s.joblib" % 
                                         (bucketID))
            pipeline = joblib.load(pipeline_path)
            feature_combiner = pipeline['encoder']
            if 'scaler' in pipeline.named_steps:
                scaler = pipeline['scaler']
            else:
                scaler = None
            cls = pipeline['cls']

            #import training data for bucket
            trainingdata = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            targets = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/y_train_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            if scaler != None:
                trainingdata = scaler.transform(trainingdata)

            #find relevant samples for bucket
            sample_instances = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID))).values
            results = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID)), sep=";")
            
            if scaler != None:
                sample_instances = scaler.transform(sample_instances)
            
            acv_explainer = joblib.load(os.path.join(PATH,'%s/%s/%s/acv_surrogate/acv_explainer_bucket_%s.joblib'% 
                                                                    (dataset_ref, cls_method, method_name, bucketID)))

            feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
#             class_names = ["Negative", "Positive"]
            
#             cats = [feat for col in dataset_manager.dynamic_cat_cols+dataset_manager.static_cat_cols 
#                     for feat in range(len(feat_list)) if col in feat_list[feat]]

            subset_stability = []
            weight_stability = []
            adjusted_weight_stability = []

            #create explainer now that can be passed later
#             lime_explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
#                                   feature_names = feat_list, class_names=class_names, categorical_features = cats)
            
            instance_no = 0
            print(len(sample_instances))
            #explain the chosen instances and find the stability score
            for instance in tqdm(sample_instances):
                instance_no += 1

                print("Testing", instance_no, "of", len(sample_instances), ".")

                #Get lime explanations for instance
                feat_pres = []
                feat_weights = []
                
               
                
                for iteration in list(range(exp_iter)):
                    weights, feat_pos = get_acv_features(acv_explainer, instance, cls, trainingdata, targets, 1)
            #        print(weights)
            #        print(feat_pos)

                    feat_pos = list(feat_pos)

                    presence_list = np.array([0]*len(feat_list))                    
                    presence_list[feat_pos] = 1

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

            results["ACV Subset Stability"] = subset_stability
            results["ACV Weight Stability"] = weight_stability
            results["ACV Adjusted Weight Stability"] = adjusted_weight_stability
            results.to_csv(os.path.join(PATH,"%s/%s/%s/samples/results_bucket_%s.csv") % 
                               (dataset_ref, cls_method, method_name, bucketID), sep=";", index=False)                
            all_results.append(results)

if xai_method=="LINDA":

    for dataset_name in datasets:
        
        num_buckets = len([name for name in os.listdir(os.path.join(PATH,'%s/%s/%s/pipelines'% 
                                                                    (dataset_ref, cls_method, method_name)))])
        dataset_manager = DatasetManager(dataset_name)

        for bucket in tqdm(range(num_buckets)):
            bucketID = bucket+1
            print ('Bucket', bucketID)
            
            #import everything needed to sort and predict
            pipeline_path = os.path.join(PATH, save_to, "pipelines/pipeline_bucket_%s.joblib" % 
                                         (bucketID))
            pipeline = joblib.load(pipeline_path)
            feature_combiner = pipeline['encoder']
            if 'scaler' in pipeline.named_steps:
                scaler = pipeline['scaler']
            else:
                scaler = None
            cls = pipeline['cls']

            #import training data for bucket
            trainingdata = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/train_data_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            targets = pd.read_csv(os.path.join(PATH, "%s/%s/%s/train_data/y_train_bucket_%s.csv" % 
                                                          (dataset_ref, cls_method, method_name, bucketID))).values
            if scaler != None:
                trainingdata = scaler.transform(trainingdata)

            #find relevant samples for bucket
            sample_instances = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/test_sample_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID))).values
            results = pd.read_csv(os.path.join(PATH, "%s/%s/%s/samples/results_bucket_%s.csv" % 
                                      (dataset_ref, cls_method, method_name, bucketID)), sep=";")
            
            if scaler != None:
                sample_instances = scaler.transform(sample_instances)

            test_dict = generate_local_predictions( sample_instances, results["Actual"], cls, scaler, None )

            feat_list = [feat.replace(" ", "_") for feat in feature_combiner.get_feature_names()]
#             class_names = ["Negative", "Positive"]
            
#             cats = [feat for col in dataset_manager.dynamic_cat_cols+dataset_manager.static_cat_cols 
#                     for feat in range(len(feat_list)) if col in feat_list[feat]]

            subset_stability = []
            weight_stability = []
            adjusted_weight_stability = []

            #create explainer now that can be passed later
#             lime_explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
#                                   feature_names = feat_list, class_names=class_names, categorical_features = cats)
            
            instance_no = 0
            print(len(sample_instances))
            #explain the chosen instances and find the stability score
            for instance in tqdm(test_dict):
                instance_no += 1

                print("Testing", instance_no, "of", len(sample_instances), ".")

                #Get lime explanations for instance
                feat_pres = []
                feat_weights = []
                
               
                
                for iteration in list(range(exp_iter)):
                    weights, feat_pos = get_linda_features(instance, cls, scaler, dataset_ref, 1, feat_list, 1)
                    #print(weights)
                    #print(feat_pos)
                    
                    feat_pos = list(feat_pos)
                    
                    bins = pd.cut(weights, 4, duplicates = "drop", retbins = True)[-1]
                    q1_min = bins[-2]

                    presence_list = np.array([0]*len(feat_list))                    

                    for n in range(len(feat_list)):
                        if weights[n] >= q1_min:
                            presence_list[n] = 1

                    feat_pres.append(presence_list)
                    feat_weights.append(weights)
                
                #print(feat_pres)
                #print(feat_weights)
                
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

            results["LINDA Subset Stability"] = subset_stability
            results["LINDA Weight Stability"] = weight_stability
            results["LINDA Adjusted Weight Stability"] = adjusted_weight_stability
            results.to_csv(os.path.join(PATH,"%s/%s/%s/samples/results_bucket_%s.csv") % 
                               (dataset_ref, cls_method, method_name, bucketID), sep=";", index=False)                
            all_results.append(results)

pd.concat(all_results).to_csv(os.path.join(PATH,"%s/%s/%s/samples/results.csv") % (dataset_ref, cls_method, method_name), 
                               sep=";", index=False)

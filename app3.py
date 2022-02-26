# coding: utf-8

import pickle

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask("__name__")

model = pickle.load(open('rf_model_final.sav', 'rb'))
df_1=pd.read_csv("data_final.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
    
    '''
    arpu_8
    onnet_mou_8	
    offnet_mou_8	
    roam_ic_mou_8	
    roam_og_mou_8	
    loc_og_t2t_mou_8	
    loc_og_t2m_mou_8	
    loc_og_t2f_mou_8	
    loc_og_t2c_mou_8	
    std_og_t2m_mou_8	
    std_og_t2f_mou_8	
    std_og_mou_8	
    isd_og_mou_8	
    spl_og_mou_8	
    og_others_8	
    total_og_mou_8	
    loc_ic_t2t_mou_8	
    loc_ic_t2m_mou_8	
    loc_ic_t2f_mou_8	
    std_ic_t2t_mou_8	             
    std_ic_t2m_mou_8	
    std_ic_t2f_mou_8	
    spl_ic_mou_8	
    isd_ic_mou_8	
    ic_others_8	
    total_rech_num_8	
    max_rech_amt_8	
    last_day_rch_amt_8	
    total_rech_data_8	
    max_rech_data_8	
    count_rech_3g_8	
    av_rech_amt_data_8	
    vol_2g_mb_8	
    vol_3g_mb_8	
    arpu_3g_8	
    night_pck_user_8	
    monthly_2g_8	
    fb_user_8	
    aon	
    aug_vbc_3g
    '''
    

    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)
    probablity = model.predict_proba(final_features)
    
    if prediction==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('index.html', output1=o1, output2=o2)
    
app.run(debug=True)

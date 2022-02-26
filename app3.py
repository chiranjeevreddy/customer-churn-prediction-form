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

@app.route('/')
def form():
    return """
        <html>
<head>
   <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
</head>
   <body>
      <title>Customer Churn Prediction</title>
	  <div class="container">
		<div class="row">

			<form action="{{ url_for('predict')}}"method="post">
    	<input type="text" name="arpu_8" placeholder="arpu_8" required="required" />
        <input type="text" name="onnet_mou_8" placeholder="onnet_mou_8" required="required" />
		<input type="text" name="offnet_mou_8" placeholder="offnet_mou_8" required="required" />
		<input type="text" name="roam_ic_mou_8" placeholder="roam_ic_mou_8" required="required" />
		<input type="text" name="roam_og_mou_8" placeholder="roam_og_mou_8" required="required" />
		<input type="text" name="loc_og_t2t_mou_8" placeholder="loc_og_t2t_mou_8" required="required" />
		<input type="text" name="loc_og_t2m_mou_8" placeholder="loc_og_t2m_mou_8" required="required" />
		<input type="text" name="loc_og_t2f_mou_8" placeholder="loc_og_t2f_mou_8" required="required" />
		<input type="text" name="loc_og_t2c_mou_8" placeholder="loc_og_t2c_mou_8" required="required" />
		<input type="text" name="std_og_t2m_mou_8" placeholder="std_og_t2m_mou_8" required="required" />
		<input type="text" name="std_og_t2f_mou_8" placeholder="std_og_t2f_mou_8" required="required" />
		<input type="text" name="std_og_mou_8" placeholder="std_og_mou_8" required="required" />
		<input type="text" name="isd_og_mou_8" placeholder="isd_og_mou_8" required="required" />
		<input type="text" name="spl_og_mou_8" placeholder="spl_og_mou_8" required="required" />
		<input type="text" name="og_others_8" placeholder="og_others_8" required="required" />
		<input type="text" name="total_og_mou_8" placeholder="total_og_mou_8" required="required" />
		<input type="text" name="loc_ic_t2t_mou_8" placeholder="loc_ic_t2t_mou_8" required="required" />
		<input type="text" name="loc_ic_t2m_mou_8" placeholder="loc_ic_t2m_mou_8" required="required" />
		<input type="text" name="loc_ic_t2f_mou_8" placeholder="loc_ic_t2f_mou_8" required="required" />
		<input type="text" name="std_ic_t2t_mou_8" placeholder="std_ic_t2t_mou_8" required="required" />
		<input type="text" name="std_ic_t2m_mou_8" placeholder="std_ic_t2m_mou_8" required="required" />
		<input type="text" name="std_ic_t2f_mou_8" placeholder="std_ic_t2f_mou_8" required="required" />
		<input type="text" name="spl_ic_mou_8" placeholder="spl_ic_mou_8" required="required" />
		<input type="text" name="isd_ic_mou_8" placeholder="isd_ic_mou_8" required="required" />
		<input type="text" name="ic_others_8" placeholder="ic_others_8" required="required" />
		<input type="text" name="total_rech_num_8" placeholder="total_rech_num_8" required="required" />
		<input type="text" name="max_rech_amt_8" placeholder="max_rech_amt_8" required="required" />
		<input type="text" name="last_day_rch_amt_8" placeholder="last_day_rch_amt_8" required="required" />
		<input type="text" name="total_rech_data_8" placeholder="total_rech_data_8" required="required" />
		<input type="text" name="max_rech_data_8" placeholder="max_rech_data_8" required="required" />
		<input type="text" name="count_rech_3g_8" placeholder="count_rech_3g_8" required="required" />
		<input type="text" name="av_rech_amt_data_8" placeholder="av_rech_amt_data_8" required="required" />
		<input type="text" name="vol_2g_mb_8" placeholder="vol_2g_mb_8" required="required" />
		<input type="text" name="vol_3g_mb_8" placeholder="vol_3g_mb_8" required="required" />
		<input type="text" name="arpu_3g_8" placeholder="arpu_3g_8" required="required" />
		<input type="text" name="night_pck_user_8" placeholder="night_pck_user_8" required="required" />
		<input type="text" name="monthly_2g_8" placeholder="monthly_2g_8" required="required" />
		<input type="text" name="fb_user_8" placeholder="fb_user_8" required="required" />
		<input type="text" name="aon" placeholder="aon" required="required" />
		<input type="text" name="aug_vbc_3g" placeholder="aug_vbc_3g" required="required" />
		




        <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
            </form>
		<div class="row"> 
			<div class="col-sm-9">
			<textarea class="form-control" rows="2" id="comment" name="query6" rows="2" cols="100" autofocus>{{output1}}</textarea>
			<textarea class="form-control" rows="2" id="comment" name="query7" rows="2" cols="100" autofocus>{{output2}}</textarea>
			</div>
		</div>
	</div>		
   </body>
</html>
    """


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

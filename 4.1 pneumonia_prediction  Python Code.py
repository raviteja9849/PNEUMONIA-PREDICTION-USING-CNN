#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 12 18:58:18 2020

@author: krishna
"""
"""Pneumonia Prediction.py"""



def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
    
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            print("prediction",preds)
            preds = preds.reshape(-1,)
        index = ['NORMAL','PNEUMONIA']
        text = ""+index[preds[0]]

    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    



















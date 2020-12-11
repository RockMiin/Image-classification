from flask import Flask, render_template, request
import urllib.request as rq
from keras.models import load_model
from PIL import Image
import numpy as np

# label
select= ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']

app= Flask(__name__)

@app.route('/')

# rendering html
def main_load():
    return render_template('web.html')

# use load_image function
@app.route('/load_image', methods= ['POST', 'GET'])
def load_image():
    if request.method=='POST':
        result= request.form['url']

        # image save
        rq.urlretrieve(result, 'image.jpg')

        # image open & resize
        img= Image.open('image.jpg')
        img= img.resize((32, 32))
        arr= np.array(img).reshape(1, 32, 32, 3)

        # load model & predict
        model= load_model('vgg_model_30.h5')
        pred = model.predict(arr)
        pred_item= np.argmax(pred)

    return render_template('web.html', result=select[pred_item], route=result)
if __name__== '__main__':
    app.run(host='127.0.0.1', debug=True)

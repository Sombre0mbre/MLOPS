import os
import zipfile
from flask import Flask, request, render_template, redirect, url_for, send_file
from model import get_prediction, train
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Setup flask
app = Flask("Our model API", template_folder='templates', static_folder='staticFiles')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        images = request.files.getlist('photos')
        # check if the post request has the file part
        if images[0].filename == '':
            return render_template('index.html', hasFile=False)
        images_urls = []
        predictions = []
        for image in images:
            image.save(os.path.join('tmp/', image.filename))
            images_urls.append(image.filename)
            predictions.append(get_prediction(image))

        return redirect(url_for('results', images=images_urls, result=predictions))
    return render_template('index.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    images = request.args.getlist('images')
    predictions = request.args.getlist('result')
    return render_template('results.html', images=zip(images, predictions))


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        zipped_data = request.files.getlist('file')
        # check if the post request has the file part
        if zipped_data[0].filename == '':
            return render_template('retrain.html', hasFile=False)
        zipped_data[0].save(os.path.join('tmp/', zipped_data[0].filename))
        with zipfile.ZipFile(f'tmp/{zipped_data[0].filename}', "r") as zip_ref:
            zip_ref.extractall("data")
        res = train(f"data/{zipped_data[0].filename.split('.')[0]}")

        return render_template('retrain.html', result=res)
    return render_template('retrain.html')


@app.route('/image/<path:filename>')
def display_image(filename):
   return send_file(f'tmp/{filename}', mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


from flask import Flask, render_template, send_file, make_response, Response
import asyncio
from functools import wraps
import threading
from flask import request
import json
import cv2
import ai_processor
import base64
import numpy as np
from model_downloaders import styles
processor = ai_processor.AIProcessor()
import cv2
image = cv2.cvtColor(
    src=cv2.imread(filename='static/sample_data/pexels-photo-16075175.jpeg'),
    code=cv2.COLOR_BGR2RGB,
)


app = Flask(__name__)

def async_action(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped


@app.route('/for_test')
def index_test():
    return render_template('test.html')

@app.route('/')
def index():
    return render_template('design.html')

@app.route('/threading')
async def temp_async():
    task_results = {'done':False, 'image':None}
    # tempThread = threading.Thread(target = processor.do_inference_async,args=(image, styles[1], 1, task_results))
    # tempThread.start()
    processor.do_inference_async(image, styles[1], 1, task_results)
    timeout = 0
    TIMEOUT_LIMIT = 10
    while not task_results['done']:
        print('task not done')
        
        timeout+=1
        print(timeout)
        if(timeout  >= TIMEOUT_LIMIT):
            break
        await asyncio.sleep(1)
    if(timeout >= TIMEOUT_LIMIT):
        print('task failed')
        return Response({'result':False})
    
    print('task done')
    #return {'result' : task_results['done']}
    retval, buffer = cv2.imencode('.png', task_results['image'])
    return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route("/upload", methods=["GET", "POST"])
async def index__():
    if(request.method == "GET"):
        return Response('fail') 
    data = request.form
    print(data['style'], data['bg'])
    img = readb64(data['image'])
    
    task_results = {'done':False, 'image':None}
    # tempThread = threading.Thread(target = processor.do_inference_async,args=(img, styles[int(data['style'])], int(data['bg']), task_results))
    # tempThread.start()
    processor.do_inference_async(img, styles[int(data['style'])], int(data['bg']), task_results)
    timeout = 0
    TIMEOUT_LIMIT = 10
    while not task_results['done']:
        print('task not done')
        timeout+=1
        if(timeout  >= TIMEOUT_LIMIT):
            break
        await asyncio.sleep(1)
    if(timeout >= TIMEOUT_LIMIT):
        print('task failed')
        return Response('fail')
    
    print('task done')
    #return {'result' : task_results['done']}
    retval, buffer = cv2.imencode('.png', task_results['image'])
    png_as_text = base64.b64encode(buffer)
    return Response(png_as_text)


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


app.run(host='0.0.0.0',port=5000)
import base64
import numpy
import json
from io import BytesIO

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

from neuralnetwork.neuralnetwork import DigitNeuralNetwork

digitNN = DigitNeuralNetwork()

@csrf_exempt
def process(request):
    imageAsString = request.body
    imageAsString = str(imageAsString).split(',')[1]
    im = Image.open(BytesIO(base64.b64decode(imageAsString))).convert('L')
    im = im.resize([8,8],Image.ANTIALIAS)
    x = [(255 - numpy.array(im.getdata())) / 255 * 16]

    response_data = {}
    response_data['result'] = str(digitNN.predict(x)[0])
    response_data['probability'] = max(digitNN.predict_proba(x)[0])
    response = HttpResponse(json.dumps(response_data), content_type="application/json")
    response['Access-Control-Allow-Origin'] = '*'
    return response

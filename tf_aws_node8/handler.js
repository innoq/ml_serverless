const path = require('path');
const os = require('os');
const fs = require('fs');
const jpeg = require('jpeg-js');
const Busboy = require('busboy');
global.fetch = require('node-fetch')

const tf = require('@tensorflow/tfjs')
const mobilenet = require('@tensorflow-models/mobilenet');
require('@tensorflow/tfjs-converter')
require('@tensorflow/tfjs-node')

const MODEL_URL = 'https://storage.googleapis.com/ml-flowers/tensorflowjs_model.pb';
const WEIGHTS_URL = 'https://storage.googleapis.com/ml-flowers/weights_manifest.json';
const CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'];
const NUMBER_OF_CHANNELS = 3

const imageByteArray = (image, numChannels) => {
  const pixels = image.data
  const numPixels = image.width * image.height;
  const values = new Float32Array(numPixels * numChannels);

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel] / 255;
    }
  }
  return values
}

const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels)
  const outShape = [image.height, image.width, numChannels];
  const input = tf.tensor3d(values, outShape, 'float32').expandDims(0);
  return input
}

var model = null

const calc = (file) => {
  return new Promise(async resolve => {
    var jpegData = fs.readFileSync(file);
    var pixels = jpeg.decode(jpegData);
    const input = imageToInput(pixels, NUMBER_OF_CHANNELS)

    if (model === null) {
      model = new tf.FrozenModel(MODEL_URL, WEIGHTS_URL)
      console.log('model loading...', Date.now())
      await model.load()
      console.log('model loaded !!!', Date.now())
    }

    console.log('classifying...', Date.now())
    predictions = await model.predict(input)
    predictions_array = Array.from(predictions.flatten().dataSync())

    const flower = CLASSES[predictions_array.indexOf(Math.max(...predictions_array))];
    console.log('classified!', flower)
    resolve(flower);
  });
}

const getContentType = (event) => {
  let contentType = event.headers['content-type']
  if (!contentType) {
    return event.headers['Content-Type'];
  }
  return contentType;
};


const headers = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'OPTIONS, POST',
  'Access-Control-Allow-Headers': 'Content-Type'
};


const parseForm = (body, headers) => new Promise((resolve, reject) => {
  const contentType = headers['Content-Type'] || headers['content-type'];
  const bb = new Busboy({ headers: { 'content-type': contentType }});
  const tmpdir = os.tmpdir();
  //const tmpdir = '/Users/mperlin/infrastructure/ml_serverless/tmp/';

  var data = {};

  const uploads = {};
  let fileWrites = [];

  bb.on('file', function (fieldname, file, filename, encoding, mimetype) {
    console.log('File [%s]: filename=%j; encoding=%j; mimetype=%j', fieldname, filename, encoding, mimetype);
    const filepath = path.join(tmpdir, filename);
    uploads[fieldname] = filepath;
    const writeStream = fs.createWriteStream(filepath);
    file.pipe(writeStream);

    file.on('data', data => console.log('File [%s] got %d bytes', fieldname, data.length))
    //.on('end', () => console.log('File [%s] Finished', fieldname));
    file.on('end', () => {
      writeStream.end();
      console.log('File [%s] Finished', fieldname)
    });
  }).on('field', (fieldname, val) => {
    data[fieldname] = val;
  }).on('finish', () => {
    for (const name in uploads) {
      file = uploads[name];
      console.log(file)
      resolve(file)
      //calc(file)
      //fs.unlinkSync(file);
    }
    //resolve(data);
  }).on('error', err => {
    reject(err);
  });

  bb.end(body);
});


exports.which_flower = async (event, context) => {
  parseForm(event.body, event.headers).then(file => calc(file))

  return {
    statusCode: 200,
    body: JSON.stringify({
      message: 'xx'
    })
  };

  // Use this code if you don't use the http event with the LAMBDA-PROXY integration
  // return { message: 'Go Serverless v1.0! Your function executed successfully!', event };
};

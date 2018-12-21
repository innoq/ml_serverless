const path = require('path');
const os = require('os');
const fs = require('fs');
const jpeg = require('jpeg-js');
const Busboy = require('busboy');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-converter');
require('@tensorflow/tfjs-node');
const shortid = require('shortid');
global.fetch = require('node-fetch');

const tfc = require('@tensorflow/tfjs-core');

const MODEL_URL = 'https://storage.googleapis.com/which_flower/tensorflowjs_model.pb';
const WEIGHTS_URL = 'https://storage.googleapis.com/which_flower/weights_manifest.json';
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
  return values;
}

const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels);
  const outShape = [image.height, image.width, numChannels];
  const input = tf.tensor3d(values, outShape, 'float32').expandDims(0);
  return input;
}

const loadModel = async () => {
  console.log('model loading...', new Date().toISOString());
  const model = new tf.FrozenModel(MODEL_URL, WEIGHTS_URL);
  await model.load();
  console.log('model loaded!', new Date().toISOString());
  return model;
}

const calc = async (file, tfModel) => {
  console.log("Decoding:", file);
  var jpegData = fs.readFileSync(file);
  var pixels = jpeg.decode(jpegData);
  const input = imageToInput(pixels, NUMBER_OF_CHANNELS);
  console.log('classifying...', new Date().toISOString());
  predictions = await tfModel.predict(input);
  predictions_array = Array.from(predictions.flatten().dataSync());
  const flower = CLASSES[predictions_array.indexOf(Math.max(...predictions_array))];
  console.log("predicted:", flower);
  return flower;
}

let model;
const parseRequest = (event) => new Promise((resolve, reject) => {
  console.log("got a request");

  const busboy = new Busboy({ headers: event.headers });
  const tmpdir = os.tmpdir();
  const uploads = {};
  const filenames = {};
  let fileWrites = [];

  busboy.on('field', function(fieldname, val) {
    const https = require("https");
    console.log(`fetching ${val}`);
    const filepath = path.join(tmpdir, shortid.generate());
    uploads[fieldname] = filepath;
    filenames[fieldname] = val;
    const file = fs.createWriteStream(filepath);
    const promise = new Promise(function(resolve, reject) {
      var req = https.get(val, response => {
          var stream = response.pipe(file);
          stream.on("finish", resolve);
          stream.on('error', reject);
      });
      req.on('error', function(err) {
        reject(err);
      });
      req.end();
    });
    fileWrites.push(promise);
  });

  busboy.on('file', (fieldname, file, filename) => {
    console.log(`Processing file ${filename}`);
    const filepath = path.join(tmpdir, shortid.generate());
    uploads[fieldname] = filepath;
    filenames[fieldname] = filename;
    const writeStream = fs.createWriteStream(filepath);
    file.pipe(writeStream);
    const promise = new Promise((resolve, reject) => {
      file.on('end', () => {
        writeStream.end();
      });
      writeStream.on('finish', resolve);
      writeStream.on('error', reject);
    });
    fileWrites.push(promise);
  });

  busboy.on('finish', () => {
    Promise.all(fileWrites)
    .then(() => Promise.all(
      Object
      .keys(uploads)
      .map(fieldname => calc(uploads[fieldname], model).then((result) => {
        try {
          fs.unlinkSync(uploads[fieldname]);
        } catch (err) {};
        return { 
            fieldname,
            filename: filenames[fieldname],
            result
        };
      }))
    )).then((results) => {
      resolve(results);
    })
    .catch( (error) => {
      console.log(error);
      reject(error);
    });
  });
  if (event.body) { busboy.write(event.body, event.isBase64Encoded ? 'base64' : 'binary');}
  busboy.end();
  /** */
});

exports.whichflower = async (event, context) => {
  console.log("starting");
  model = model || await loadModel();
  const result = await parseRequest(event);
  console.log("result", result);
  return {
    statusCode: 200,
    body: JSON.stringify(result)
  };
};

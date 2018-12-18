const path = require('path');
const os = require('os');
const fs = require('fs');
const jpeg = require('jpeg-js');
const Busboy = require('busboy');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-converter');
require('@tensorflow/tfjs-node');
global.fetch = require('node-fetch');
const shortid = require('shortid');

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
  const result = await model.load();
  console.log(result);
  console.log('model loaded!', new Date().toISOString());
  return model;
}

const calc = async (file, tfModel) => {
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
const upload_file = async (req, res) => {
  if (!model) {
    model = await loadModel();
  } else {
    console.log("model already available")
  }
  if (req.method === 'POST') {
    console.log("got a POST request");
    const busboy = new Busboy({ headers: req.headers });
    const tmpdir = os.tmpdir();
    const uploads = {};
    const filenames = {};
    let fileWrites = [];

    busboy.on('field', function(fieldname, val, fieldnameTruncated, valTruncated, encoding, mimetype) {
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
      ))
      .then((results) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write(JSON.stringify(results));
        res.end();
      })
      .catch( (error) => {
        console.log(error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.write(JSON.stringify({ error: error || "Error" }));
        res.end();
      });
    });

    // This allows to run locally inside a plain old nodejs http server
    req.rawBody ? busboy.end(req.rawBody) : req.pipe(busboy);
  } else {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.write(JSON.stringify({ error: "Error" }));
    res.end();
  }
};

exports.gcloud_tensorflow_node8_upload = (req, res) => {
  upload_file(req, res);
}

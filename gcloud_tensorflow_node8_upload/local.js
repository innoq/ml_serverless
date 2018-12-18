var http = require('http');
var upload_file = require('./script.js').which_flower_debug;
http.createServer((req, res) => {
    upload_file(req, res);
  }).listen(8080);
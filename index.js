var express = require('express');
var bodyParser = require('body-parser');
var morgan = require('morgan');
var http = require('http');
var fs = require('fs');
//var mongoose = require('mongoose');
var config = require('./config');
//var db = require('./app/db');
var cifar10 = require('./app/models/cifar10');
var mnist = require('./app/models/mnist');

var app = express();
app.use(bodyParser.json({limit: '10mb'})); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' })); // support encoded bodies

//var user = new db.create_user('Roman', 'Kaplan', "some@one.com");
//console.log("created user:" + user.firstname);
//db.add_user(user);
//db.find_user('Roman');

// Serve HTTP requests
app.set('port', (process.env.PORT || 8080));
app.use(express.static(__dirname + '/public'));

app.get('/', function(request, response) {
    //fs.readFileSync("index.html");
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});

// MNIST
app.get('/train_mnist', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/mnist/train_mnist.html"))
    response.send(index_buffer.toString())
});

app.get('/test_mnist', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/mnist/test_mnist.html"))
    response.send(index_buffer.toString())
});

// CIFAR10
app.get('/train_cifar10', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/cifar10/train_cifar10.html"))
    response.send(index_buffer.toString())
});

app.get('/test_cifar10', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/cifar10/test_cifar10.html"))
    response.send(index_buffer.toString())
});

app.get('/get_init_model_from_server', function(request, response){
    if (request.query.model_name === "CIFAR10")
        response.send(cifar10.init_model)
    else if (request.query.model_name === "MNIST")
        response.send(mnist.init_model)
    else
        console.log(" <get_init_model_from_server> Received unknown model request: " + request.query.model_name);
});

app.get('/test_model_from_server', function(request, response){
    console.log("Received a test request");
    //Send net.JSON to client + test batch name
});

app.post('/store_model_on_server', function(request, response){
    console.log("<store_model_on_server()> model_name: " + request.body.model_name);
    console.log("<store_model_on_server()> net (JSON) size: " + request.body.net.length);

    console.log("cifar10 net len: " + cifar10.net.toJSON());

    response.send("POST handled in Node.js server")
});

app.listen(app.get('port'), function() {
  console.log("Node app is running at localhost:" + app.get('port'))
});
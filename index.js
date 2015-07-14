var express = require('express');
var bodyParser = require('body-parser');
var compress = require('compression');
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
app.use(compress());

//var user = new db.create_user('Roman', 'Kaplan', "some@one.com");
//console.log("created user:" + user.firstname);
//db.add_user(user);
//db.find_user('Roman');

// Serve HTTP requests
app.set('port', (process.env.PORT || 8080));
app.use(express.static(__dirname + '/public'));


/////// =====================

app.get('/', function(request, response) {
    //fs.readFileSync("index.html");
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});

// ========  MNIST ========
app.get('/train_mnist', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/mnist/train_mnist.html"))
    response.send(index_buffer.toString())
});

app.get('/test_mnist', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/mnist/test_mnist.html"))
    response.send(index_buffer.toString())
});

// ======== CIFAR10 ========
app.get('/train_cifar10', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/cifar10/train_cifar10.html"))
    response.send(index_buffer.toString())
});

app.get('/test_cifar10', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/cifar10/test_cifar10.html"))
    response.send(index_buffer.toString())
});


//==========================================
//=  Store and load models to / from server
//==========================================

app.get('/get_init_model_from_server', function(request, response){
    if (request.query.model_name === "CIFAR10") {
        response.send(cifar10.init_model)
    }
    else if (request.query.model_name === "MNIST") {
        response.send(mnist.init_model)
    }
    else {
        console.log(" <get_init_model_from_server> Received unknown model request: " + request.query.model_name);
    }
});

app.get('/get_net_and_batch_from_server', function(request, response){
    if (request.query.model_name === "CIFAR10") {
        parameters = {net : cifar10.net_manager.get_weights(), batch_num: cifar10.net_manager.get_and_update_batch_num()};
        //parameters = {net : cifar10.net_manager.get_weights()};
        //console.log(" <get_init_model_from_server> Sending the following net after `stringify`: " + parameters.net.substring(0, 1000));
        response.send(parameters);
    }
    else if (request.query.model_name === "MNIST") {
        parameters = {net : mnist.net_manager.get_weights()};
        //parameters = {net : mnist.net_manager.get_weights()};
        //console.log(" <get_init_model_from_server> Sending the following net after `stringify`: " + parameters.net.substring(0, 1000));
        response.send(parameters);
    }
    else {
        console.log(" <get_init_model_from_server> Received unknown model request: " + request.query.model_name);
    }
});

app.get('/get_batch_num_from_server', function(request, response) {
    var batch_num = cifar10.net_manager.get_batch_num();
    parameters = { batch_num: batch_num};
    response.send(parameters);
});

app.get('/test_model_from_server', function(request, response){
    console.log("Received a test request");
    //Send net.JSON to client + test batch name
});

app.post('/store_weights_on_server', function(request, response){
    //Expecting to receive JSON of the form: {model_name: <model name>, net: <net in JSON>}
    var model_name = request.body.model_name;
    console.log("<store_weights_on_server()> model_name: " + model_name);
    console.log("<store_weights_on_server()> net (in JSON) size: " + request.body.net.length);
    //console.log("<store_weights_on_server()> Received: " + request.body.net.substring(0, 1000));

    if (request.body.model_name === "CIFAR10") {
        cifar10.net_manager.store_weights(request.body.net);

        //DEBUG - testing to see if net is stored properly
        console.log("<store weights on server>: Checking if net is stored properly");
        if (request.body.net === cifar10.net_manager.get_weights()) {
            console.log("<store weights on server>: SUCCESS!!! - weights are stored properly");
        }
        else {
            console.log("<store weights on server>: Failure... weights aren't stored properly");
        }
    }
    else if (request.body.model_name === "MNIST") {
        mnist.net_manager.store_weights(request.body.net);
    }
    else {
        var new_model_weights = request.body.net;
        response.send("Received unknown-model weights on Node.js server");
        //do something with the weights
    }
    response.send("Stored " + model_name + " weights on Node.js server");
});

// ====== Default case ========

app.get('*', function(request, response) {
    //fs.readFileSync("index.html");
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});


app.listen(app.get('port'), function() {
  console.log("Node app is running at localhost:" + app.get('port'))
});
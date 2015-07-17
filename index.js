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

var curr_model_ID = 0;

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

// ======== CIFAR10 ========
app.get('/train', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/cifar10/train.html"))
    response.send(index_buffer.toString())
});

app.get('/admin', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/admin.html"))
    response.send(index_buffer.toString())
});

//==========================================
//=  Store and load models to / from server
//==========================================

app.get('/get_init_model_from_server', function(request, response){
    var params = {init_model: cifar10.init_model, net : cifar10.net_manager.get_weights()};
    response.send(params);
});

app.get('/get_net_and_update_batch_from_server', function(request, response){
    model_parameters = cifar10.net_manager.get_model_parameters();
    var parameters = {net : cifar10.net_manager.get_weights(), batch_num: cifar10.net_manager.get_and_update_batch_num(), model_ID: curr_model_ID,
                        learning_rate : model_parameters.learning_rate , momentum : model_parameters.momentum , l2_decay: model_parameters .l2_decay};
    //parameters = {net : cifar10.net_manager.get_weights()};
    //console.log(" <get_net_and_update_batch_from_server> Sending the following net after `stringify`: " + parameters.net.substring(0, 1000));
    response.send(parameters);
});


app.get('/get_net_and_batch_from_server', function(request, response){
    model_parameters = cifar10.net_manager.get_model_parameters();
    var parameters = {net : cifar10.net_manager.get_weights(), batch_num: cifar10.net_manager.get_batch_num(), model_ID: curr_model_ID,
        learning_rate : model_parameters.learning_rate , momentum : model_parameters.momentum , l2_decay: model_parameters .l2_decay};
    //parameters = {net : cifar10.net_manager.get_weights()};
    //console.log(" <get_net_and_batch_from_server> Sending the following net after `stringify`: " + parameters.net.substring(0, 1000));
    response.send(parameters);
});


app.get('/get_batch_num_from_server', function(request, response) {
    var batch_num = cifar10.net_manager.get_batch_num();
    var parameters = { batch_num: batch_num};
    response.send(parameters);
});

app.post('/update_model_from_gradients', function(request, response){
    //Expecting to receive JSON of the form: {model_name: <model name>, net: <net in JSON>}
    var model_name = request.body.model_name;
    console.log("<store_weights_on_server()> model_name: " + model_name);
    console.log("<store_weights_on_server()> net (in JSON) size: " + request.body.net.length);
    //console.log("<store_weights_on_server()> Received: " + request.body.net.substring(0, 1000));

    cifar10.net_manager.update_model_from_gradients(request.body);

    response.send("Stored " + model_name + " weights on Node.js server");
});

app.post('/reset_model', function(request, response){
    cifar10.reset_model();
    response.send("Model was " + request.body.model_name + " resetted and initialized from 'init_model'");
    console.log("Model was " + request.body.model_name + " resetted and initialized from 'init_model'");
});

app.post('/store_new_model_on_server', function(request, response){
    cifar10.store_new_model(request.body.net);
    response.send("Model " + request.body.model_name + " was changed and saved");
    console.log("Model " + request.body.model_name + " was changed and saved");
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
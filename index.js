var express = require('express');
var bodyParser = require('body-parser');
var compress = require('compression');
var morgan = require('morgan');
var http = require('http');
var fs = require('fs');
var config = require('./config');

var DEFAULT_DATASET = "cifar10";

// Init. default net
var net_manager = require('./app/net_manager');
net_manager.set_dataset(DEFAULT_DATASET);

var node_count = 0;

var app = express();
app.use(bodyParser.json({limit: '10mb'})); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' })); // support encoded bodies
app.use(compress());

// Serve HTTP requests
app.set('port', (process.env.PORT || 8080));
app.use(express.static(__dirname + '/public'));

/////// =======  Accessible pages  =========

app.get('/', function(request, response) {
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});

app.get('/train', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/train.html"))
    response.send(index_buffer.toString())
});

app.get('/admin', function(request, response){
    var index_buffer = new Buffer(fs.readFileSync("public/admin.html"))
    response.send(index_buffer.toString())
});


/////// =======  Client status API  =========

app.get('/get_node_name', function(request, response) {
  response.send((++node_count).toString())
});

//==========================================
//=  Store and load models to / from server
//==========================================

app.get('/get_base_model_from_server', function(request, response) {
    response.send(net_manager.get_base_model());
    /*  Returns: {
          base_model  = base_model;
          id          = model_ID;
          dataset     = dataset_ID
        };
    */
});

app.get('/get_net_batch_all', function(request, response) {   //  New worker node
    var model_parameters = net_manager.get_model_parameters();
    var batch_num = net_manager.request_batch_num(request.query.client_ID);
    var parameters = {
                        net: net_manager.get_net().toJSON(),
                        batch_num: batch_num,
                        batch_size:  net_manager.get_batch_size(),
                        model_ID: net_manager.get_model_ID(),
                        trainer_param:  net_manager.get_trainer_param(),
                        batch_url:  net_manager.get_batch_url(batch_num),
                        model_name: net_manager.get_dataset_name()
                     };
    console.log(" <get_net_batch_all> Sending batch_num: " + parameters.batch_num + " to client: " + request.query.client_ID);
    response.send(parameters);
});


app.get('/get_net_and_batch_from_server', function(request, response){
    var model_parameters = cifar10.net_manager.get_model_parameters();
    var parameters = {  net : cifar10.net_manager.get_weights(),
                        batch_num: cifar10.net_manager.get_batch_num(),
                        model_ID: cifar10.net_manager.get_model_ID(),
                        learning_rate : model_parameters.learning_rate,
                        momentum : model_parameters.momentum ,
                        l2_decay: model_parameters.l2_decay
                      };
    response.send(parameters);
    console.log(" <get_net_and_batch_from_server> sent net with model_ID: " + parameters.model_ID + " to Admin");
});


app.get('/get_batch_num_from_server', function(request, response) {
    var batch_num = cifar10.net_manager.get_batch_num();
    var parameters = { batch_num: batch_num };
    response.send(parameters);
});

app.post('/update_model_from_gradients', function(request, response) {
    var model_ID_from_client = request.body.model_ID;

    if (model_ID_from_client == net_manager.get_model_ID()) {

        console.log("<store_weights_on_server()> net (in JSON) size: " + request.body.net.length);

        net_manager.update_model_from_gradients(request.body);
        response.send();
    }
    else {
        response.send("<update_model_from_gradients> Old model_ID, gradients were discarded ");
        console.log("<update_model_from_gradients> Received results from an old model_ID " + model_ID_from_client + ", discarding...");
    }
});

app.post('/reset_model', function(request, response){
    cifar10.reset_model();
    var new_model_ID = cifar10.net_manager.get_model_ID();
    response.send("Model was " + request.body.model_name + " resetted. New model_ID: " + new_model_ID);
    console.log("<reset_model> Resetting the net to:\n" + cifar10.net_manager.get_init_model());
    console.log("<reset_model> Model was " + request.body.model_name + " resetted. New model_ID: " + new_model_ID);
});

app.post('/store_new_model_on_server', function(request, response){
    cifar10.init_new_model(request.body.new_init_model);
    var new_model_ID = cifar10.net_manager.get_model_ID();
    response.send("Model " + request.body.model_name + " was changed and saved. New model_ID: " + new_model_ID);

    console.log("<store_new_model_on_server> Received new_init_model " + request.body.new_init_model);
    console.log("<store_new_model_on_server> Model " + request.body.model_name + " was changed and saved. New model_ID: " + new_model_ID);
});

// ====== Default case ========

app.get('*', function(request, response) {
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});

app.listen(app.get('port'), function() {
  console.log("Node app is running at localhost:" + app.get('port'))
});

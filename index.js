var express = require('express');
var bodyParser = require('body-parser');
var compress = require('compression');
var morgan = require('morgan');
var http = require('http');
var fs = require('fs');
var config = require('./config');

var DEFAULT_DATASET = "mnist";

// Init. default net
var net_manager = require('./app/net_manager');
net_manager.set_dataset(DEFAULT_DATASET);

var node_count = 0;

var is_model_in_testing_mode = false;

var app = express();
app.use(bodyParser.json({limit: '10mb'})); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' })); // support encoded bodies
app.use(compress());


var get_network_stats = function() {
    var stats_to_send = {fw_times_average: net_manager.get_fw_timings_average(),
        bw_times_average: net_manager.get_fw_timings_average(),
        average_latency_to_server: net_manager.get_latencies_to_server_average(),
        average_latency_from_server: net_manager.get_latencies_from_server_average(),
    };
    return stats_to_send;
}


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



app.get('/get_node_name', function(request, response) {		// Unused
  response.send((++node_count).toString())
});


app.get('/get_admin_batch', function(request, response) {		// Unused
  response.send(net_manager.gen_admin_batch_url())
});

//==========================================
//=  Store and load models to / from server
//==========================================

app.get('/get_base_model_from_server', function(request, response) {
    response.send(net_manager.get_base_model_data());
    /*  Returns: {
          base_model  = base_model;
          id          = model_ID;
          dataset     = dataset_ID
		  and various data
        };
    */
});

app.get('/get_net_batch_all', function(request, response) {   //
    var model_parameters = net_manager.get_model_parameters();
	var epoch_to_send = net_manager.get_epochs_count();
    var batch_num = net_manager.request_batch_num(request.query.client_ID);
    var parameters = {
                        net: net_manager.get_net().toJSON(),
                        batch_num: batch_num,
                        batch_size:  net_manager.get_batch_size(),
						            epoch_num: epoch_to_send,
                        model_ID: net_manager.get_model_ID(),
                        trainer_param:  net_manager.get_trainer_param(),
                        batch_url:  net_manager.get_batch_url(batch_num),
                        dataset_name: net_manager.get_dataset_name(),
						            total_different_clients: net_manager.get_different_clients_num(),
                     };
    console.log(" <get_net_and_batch_from_server> Sending batch_num: " + parameters.batch_num + " to client: " + request.query.client_ID);
    response.send(parameters);
    net_manager.add_latencies_from_server(request.query.latency_from_server);
});

// To be used by the Admin
app.get('/get_net_and_current_training_batch_from_server', function(request, response){
    if (net_manager.is_need_to_send_net_for_testing(request.query.model_ID, request.query.epoch_num)) {
        var parameters = {	net : net_manager.get_net().toJSON(),
              							batch_num: net_manager.get_batch_num(),
              							epoch_num: net_manager.get_epochs_count(),
              							model_ID: net_manager.get_model_ID(),
              							model_param: net_manager.get_model_parameters(),
              							total_different_clients: net_manager.get_different_clients_num(),
              							last_contributing_client: net_manager.get_last_contributing_client()};

        console.log(" <get_net_and_current_training_batch_from_server> sending net with model_ID " + parameters.model_ID +
                        " and in epoch_num " + parameters.epoch_num + " to Admin");
    }
    else {
        var parameters = {	batch_num: net_manager.get_batch_num(),
              							epoch_num: net_manager.get_epochs_count(),
              							model_ID: net_manager.get_model_ID(),
              							total_different_clients: net_manager.get_different_clients_num(),
              							last_contributing_client: net_manager.get_last_contributing_client()};

        console.log(" <get_net_and_current_training_batch_from_server> NOT sending net to Admin. Model_ID " + parameters.model_ID +
                        " & epoch_num " + parameters.epoch_num + " didn't update");
    }
    response.send(parameters);
});


app.get('/get_average_stats', function(request, response) {
    var stats = get_network_stats();

    response.send(stats);
    console.log("<get_stats> sent stats to requester");
});

app.get('/get_all_stats', function(request, response) {
    var stats_in_csv = net_manager.get_stats_in_csv();

    response.send(stats_in_csv);
    console.log("<get_all_stats> all sent stats to requester in CSV");
});

app.get('/get_batch_num_from_server', function(request, response) {
    var batch_num = net_manager.get_batch_num();
    var parameters = { batch_num: batch_num };
    response.send(parameters);
});


app.post('/update_model_from_gradients', function(request, response) {
    var model_ID_from_client = request.body.model_ID;

    if (model_ID_from_client == net_manager.get_model_ID()) {

        console.log("<store_weights_on_server()> net (in JSON) size: " + request.body.net.length);

        net_manager.update_model_from_gradients(request.body);
        response.send();
		net_manager.update_stats(request.body);
    }
    else {
		 if (is_model_in_testing_mode) {
            response.send("<update_model_from_gradients> Server in testing mode, stopped updating the model ");
            console.log("<update_model_from_gradients> Server in testing mode, stopped updating the model ");
        }
        else {
            response.send("<update_model_from_gradients> Old model_ID, gradients were discarded ");
            console.log("<update_model_from_gradients> Received results from an old model_ID " + model_ID_from_client + ", discarding...");
        }
    }
});

app.post('/reset_model', function(request, response){

    net_manager.reset_model();
    var new_model_ID = net_manager.get_model_ID();
    response.send("Model was " + request.body.model_name + " resetted. New model_ID: " + new_model_ID);
    console.log("<reset_model> Model was " + request.body.model_name + " resetted. New model_ID: " + new_model_ID);
});

app.post('/store_new_model_on_server', function(request, response){
    net_manager.set_net(request.body.new_init_model);
    var new_model_ID = cifar10.net_manager.get_model_ID();
    response.send("Model " + request.body.model_name + " was changed.");
    console.log("<store_new_model_on_server> Net was changed " + request.body.new_init_model);
});

app.post('/store_validation_accuracy_on_server', function(request, response){
    if (net_manager.is_new_validation_accuracy_worse(request.body.validation_accuracy, request.body.epoch_num)
            && request.body.epoch_num > cifar10.minimum_epochs_to_train) {
        var res = {is_testing_needed: true};
        response.send(res);
        is_model_in_testing_mode = true;
        console.log("<store_validation_accuracy_on_server> Received new validation accuracy: "
            + request.body.validation_accuracy + "==> +++ Going to TESTING mode");
    }
    else {
        var res = {is_testing_needed: false};
        response.send(res);
        console.log("<store_validation_accuracy_on_server> Received new validation accuracy: "
            + request.body.validation_accuracy + "==> Staying in validation mode");
    }
});


app.get('/get_validation_net', function(request, response) {

    var parameters = {	net : net_manager.get_net(),

						batch_num: net_manager.get_batch_num(),
						epoch_num: net_manager.get_epochs_count(),
						model_ID: net_manager.get_model_ID(),
            model_param: net_manager.get_model_parameters(),
						total_different_clients: net_manager.get_different_clients_num(),
						last_contributing_client: net_manager.get_last_contributing_client()
					};
    response.send(parameters);
});

app.get('/get_net_snapshot', function(request, response) {
    var net_in_JSON_to_send = net_manager.get_net().toJSON();

    response.send(net_in_JSON_to_send);
});

app.post('/change_dataset', function(request, response) {
    net_manager.set_dataset(request.body.name);
    net_manager.reset_model();

    response.send("Crazy");
});


// ====== Default case ========

app.get('*', function(request, response) {
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});

app.listen(app.get('port'), function() {
  console.log("Node app is running at localhost:" + app.get('port'))
});

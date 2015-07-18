var layer_defs, net, trainer;
var old_net, curr_net;
var curr_model_ID, curr_batch_num;

// ------------------------
// BEGIN CIFAR10 SPECIFIC STUFF
// ------------------------
var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

var use_validation_data = true;
var first_execution = true;
var validation_frequency = 2 * 1000;
var prediction_interval;
var get_validations = false;

/*$(function ()
{
    $.import_js('client.js');
});*/

// int main
$(window).load(function() {
    var AJAX_init_parameters = {model_name: "CIFAR10" };
    $.get('/get_init_model_from_server', AJAX_init_parameters, function(data) {
        console.log("Received an init_model from server: \n" + data.init_model);

        init_model = data.init_model;
        $("#newnet").val(init_model);
        eval(init_model);
        //net.fromJSON(JSON.parse(data.net));
        reset_all();
        update_net_param_display();

        for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

        load_data_batch(0); // async load train set batch 0 (6 total train batches)
        load_data_batch(test_batch); // async load test set (batch 6)
        start_fun();
    });
});

var test_prediction_accuracy = function () {
    get_net_and_batch_from_server();
}
if (get_validations)
    validation_interval = setInterval(test_prediction_accuracy, validation_frequency);

///// Communicating with the server

var toggle_validate = function () {
    var btn = document.getElementById('toggle-validate-btn');
    get_validations = !get_validations;
    if (get_validations) {
        validation_interval = setInterval(test_prediction_accuracy, validation_frequency);
        btn.value = 'Stop Validating';
    }
    else {
        clearInterval(validation_interval);
        btn.value = 'Start Validating';
    }
}

var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        curr_batch_num = data.batch_num;
        return data.batch_num;
    });
}
/*
var get_net_and_batch_from_server = function() {
    var parameters = {model_name: "CIFAR10"};
    var batch_num;
    $.get('/get_net_and_batch_from_server', parameters, function(data) {
        console.log("<get_net_and_batch_from_server> Received " + parameters.model_name + " net back");
        console.log("<get_net_and_batch_from_server> Working on batch: " + data.batch_num); //DEBUG
        console.log("<get_net_and_batch_from_server> Received " + data.net.length + " net in length back"); //DEBUG
        console.log("<get_net_and_batch_from_server> Received the NET" + data.net.substring(0,1000)); //DEBUG
        //console.log("<get_net_from_server> Received the net: " + data.net);
        batch_num = data.batch_num;
        update_displayed_batch_num(batch_num);

        old_net = net.toJSON();
        net = new convnetjs.Net();
        net.fromJSON(JSON.parse(data.net));
        reset_all();
        batch_num = data.batch_num;

        var vis_elt = document.getElementById("visnet");
        visualize_activations(net, vis_elt);
        test_predict();
        update_net_param_display();
    });
    return batch_num;
}*/

var get_net_and_batch_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_net_and_batch_from_server', parameters, function(data) {
        curr_model_ID  = data.model_ID;
        console.log("<get_net_and_batch_from_server> Received "+ parameters.model_name + " model with model_ID: " + curr_model_ID);

        var net_in_Json = JSON.parse(data.net);
        net = new convnetjs.Net();
        net.fromJSON(net_in_Json);
        reset_all();

        curr_batch_num = data.batch_num;
        update_displayed_batch_num(curr_batch_num);

        var vis_elt = document.getElementById("visnet");
        visualize_activations(net, vis_elt);
        test_predict();
        update_net_param_display();
    });
}

var reset_model = function() {
    var parameters = {model_name: "CIFAR10"};

    $.post('/reset_model', parameters, function(data) {
        console.log("Resetting the model named: <" + parameters.model_name + "> stored on server");
    });
}

var change_net = function() {
    var new_init_model = $("#newnet").val();
    //rand = get_random_number();
    var parameters = {model_name: "CIFAR10", new_init_model: new_init_model};
    console.log("<change_net> Sending the following new CIFAR10 init_model: " + new_init_model);
    reset_all();
    $.post('/store_new_model_on_server', parameters, function(data) {
        console.log(data);
    });
}
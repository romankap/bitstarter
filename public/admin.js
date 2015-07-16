var layer_defs, net, trainer;
var old_net, curr_net;
// ------------------------
// BEGIN CIFAR10 SPECIFIC STUFF
// ------------------------
var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

var use_validation_data = true;
var first_execution = true;

/*$(function ()
{
    $.import_js('client.js');
});*/

// int main
$(window).load(function() {
    var AJAX_init_parameters = {model_name: "CIFAR10" };
    $.get('/get_init_model_from_server', AJAX_init_parameters, function(data) {
        console.log("Received an init_model from server: \n" + data);

        init_model = data;
        $("#newnet").val(init_model);
        eval(init_model);
        update_net_param_display();

        for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

        load_data_batch(0); // async load train set batch 0 (6 total train batches)
        load_data_batch(test_batch); // async load test set (batch 6)
        start_fun();
    });
});

var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        return data.batch_num;
    });
}
var get_net_and_batch_from_server = function() {
    var parameters = {model_name: "CIFAR10"};
    var batch_num;
    $.get('/get_net_and_batch_from_server', parameters, function(data) {
        console.log("<get_net_and_batch_from_server> Received " + parameters.model_name + " net back");
        console.log("<get_net_and_batch_from_server> Working on batch: " + data.batch_num); //DEBUG
        console.log("<get_net_and_batch_from_server> Received " + data.net.length + " net in length back"); //DEBUG
        console.log("<get_net_and_batch_from_server> Received the NET" + data.net.substring(0,1000)); //DEBUG
        //console.log("<get_net_from_server> Received the net: " + data.net);

        old_net = net.toJSON();
        net = new convnetjs.Net();
        net.fromJSON(JSON.parse(data.net));
        reset_all();
        batch_num = data.batch_num;
    });
    return batch_num;
}

var get_model_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/test_model_from_server', parameters, function(data) {
        console.log(data);
        //var json = JSON.parse(data);
        //net = new convnetjs.Net();
        //net.fromJSON(json);
        //reset_all();
    });
}
var reset_model = function() {
    var parameters = {model_name: "CIFAR10"};

    $.post('/reset_model', parameters, function(data) {
        console.log("Resetting the model named: <" + parameters.model_name + "> stored on server");
    });
}

var change_net = function() {
    eval($("#newnet").val());
    reset_all();
    curr_net = net.toJSON();
    net_in_JSON_string = JSON.stringify(curr_net);
    rand = get_random_number();
    var parameters = {model_name: "CIFAR10",model_ID : rand, net: net_in_JSON_string };
    console.log("Sending CIFAR10 net_in_JSON with length " + parameters.net.length);
    console.log("Sending CIFAR10 net: " + parameters.net.substring(0,1000));
    $.post('/store_new_model_on_server', parameters, function(data) {
        console.log(data);
    });
}
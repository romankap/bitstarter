var layer_defs, net, trainer;
var old_net, curr_net;
var gradients_calculator = {
    traverse: function (new_net_property, old_net_property) {
        debug_global_counter = 0;

        for (var i in new_net_property) {
            if (new_net_property[i] !== null && typeof(new_net_property[i]) == "object") {
                //going on step down in the object tree!!
                this.traverse(new_net_property[i], old_net_property[i]);
            }
            else if (new_net_property[i] !== null && typeof(new_net_property[i]) !== "object" &&
                    old_net_property[i] !== null && typeof(old_net_property[i]) !== "object" &&
                    isNumeric(i)) {
                new_net_property[i] -= old_net_property[i];
            }
        }
    }
}

var debug_global_counter;
// ------------------------
// BEGIN CIFAR10 SPECIFIC STUFF
// ------------------------
var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

var use_validation_data = true;

function isNumeric(num) {
    return !isNaN(num)
}

var init_model;

// int main
$(window).load(function() {
    var AJAX_init_parameters = {model_name: "CIFAR10" };
    $.get('/get_init_model_from_server', AJAX_init_parameters, function(data) {
        console.log("Received an init_model from server: \n" + data);

        init_model = data;
        eval(init_model);
        update_net_param_display();

        for(var k=0;k<loaded.length;k++) { loaded[k] = false; }

        load_data_batch(0); // async load train set batch 0 (6 total train batches)
        load_data_batch(test_batch); // async load test set (batch 6)
        start_working();
    });
});

var start_working = function() {
    if(loaded[0] && loaded[test_batch]) {
        console.log('Good to go!');
        setInterval(load_and_step, 0); // lets go!
    }
    else { setTimeout(start_fun, 200); } // keep checking
}


// loads a training image and trains on it with the network
var paused = true;

var compute = function() {
    paused = !paused;
    var btn = document.getElementById('buttontp');
    if (paused) {
        btn.value = 'compute'
        post_net_to_server();
    }
    else {
        btn.value = 'pause';
        get_net_and_batch_from_server();
    }
}


var calculate_gradients = function() {
    gradients_calculator.traverse(curr_net, old_net, "");

    console.log("<====== After calculating gradients> NEW net: " + JSON.stringify(curr_net).substring(0, 1000));
}

var post_net_to_server = function() {
    curr_net = net.toJSON();
    calculate_gradients();
    net_in_JSON_string = JSON.stringify(curr_net);
    var parameters = {model_name: "CIFAR10", net: net_in_JSON_string };
    console.log("Sending CIFAR10 net_in_JSON with length " + parameters.net.length);
    console.log("Sending CIFAR10 net: " + parameters.net.substring(0,1000));
    $.post('/store_weights_on_server', parameters, function(data) {
        //console.log(data);
    });
}
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
        //reset_all();
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

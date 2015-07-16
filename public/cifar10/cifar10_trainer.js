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

var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var testAccWindow = new cnnutil.Window(50, 1);
var step_num = 0;
var step = function(sample) {

    var x = sample.x;
    var y = sample.label;

    if(sample.isval) {
        // use x to build our estimate of validation error
        net.forward(x);
        var yhat = net.getPrediction();
        var val_acc = yhat === y ? 1.0 : 0.0;
        valAccWindow.add(val_acc);
        return; // get out
    }

    // train on it with network
    var stats = trainer.train(x, y);
    var lossx = stats.cost_loss;
    var lossw = stats.l2_decay_loss;

    // keep track of stats such as the average training error and loss
    var yhat = net.getPrediction();
    var train_acc = yhat === y ? 1.0 : 0.0;
    xLossWindow.add(lossx);
    wLossWindow.add(lossw);
    trainAccWindow.add(train_acc);

    // visualize training status
    var train_elt = document.getElementById("trainstats");
    train_elt.innerHTML = '';
    var t = 'Forward time per example: ' + stats.fwd_time + 'ms';
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Backprop time per example: ' + stats.bwd_time + 'ms';
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Classification loss: ' + f2t(xLossWindow.get_average());
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'L2 Weight decay loss: ' + f2t(wLossWindow.get_average());
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Training accuracy: ' + f2t(trainAccWindow.get_average());
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Validation accuracy: ' + f2t(valAccWindow.get_average());
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Examples seen: ' + step_num;
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));

    // visualize activations
    if(step_num % 100 === 0) {
        var vis_elt = document.getElementById("visnet");
        visualize_activations(net, vis_elt);
    }

    // log progress to graph, (full loss)
    if(step_num % 200 === 0) {
        var xa = xLossWindow.get_average();
        var xw = wLossWindow.get_average();
        if(xa >= 0 && xw >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
            lossGraph.add(step_num, xa + xw);
            lossGraph.drawSelf(document.getElementById("lossgraph"));
        }
    }

    // run prediction on test set
    if((step_num % 100 === 0 && step_num > 0) || step_num===100) {
        test_predict();
        // post gradients to server
        post_gradients_to_server();
        get_net_and_update_batch_from_server();
    }
    step_num++;
}


// loads a training image and trains on it with the network
var paused = true;

var compute = function() {
    paused = !paused;
    var btn = document.getElementById('buttontp');
    if (paused) {
        btn.value = 'compute';
        post_gradients_to_server();
    }
    else {
        btn.value = 'pause';
        get_net_and_update_batch_from_server();
    }
}


var calculate_gradients = function() {
    gradients_calculator.traverse(curr_net, old_net, "");

    console.log("<====== After calculating gradients> NEW net: " + JSON.stringify(curr_net).substring(0, 1000));
}

var post_gradients_to_server = function() {
    curr_net = net.toJSON();
    calculate_gradients();
    net_in_JSON_string = JSON.stringify(curr_net);
    var learning_rate = trainer.learning_rate;
    var momentum = trainer.momentum;
    var l2_decay = trainer.l2_decay;
    var parameters = {model_name: "CIFAR10", net: net_in_JSON_string,
                      learning_rate :learning_rate, momentum: momentum, l2_decay: l2_decay };
    console.log("Sending CIFAR10 net_in_JSON with length " + parameters.net.length);
    console.log("Sending CIFAR10 net: " + parameters.net.substring(0,1000));
    $.post('/update_model_from_gradients', parameters, function(data) {
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
var get_net_and_update_batch_from_server = function() {
    var parameters = {model_name: "CIFAR10"};
    var batch_num;
    $.get('/get_net_and_update_batch_from_server', parameters, function(data) {
        console.log("<get_net_and_update_batch_from_server> Received " + parameters.model_name + " net back");
        console.log("<get_net_and_update_batch_from_server> Working on batch: " + data.batch_num); //DEBUG
        console.log("<get_net_and_update_batch_from_server> Received " + data.net.length + " net in length back"); //DEBUG
        console.log("<get_net_and_update_batch_from_server> Received the NET" + data.net.substring(0,1000)); //DEBUG
        //console.log("<get_net_from_server> Received the net: " + data.net);

        trainer.learning_rate = data.learning_rate;
        trainer.momentum = data.momentum;
        trainer.l2_decay = data.l2_decay;

        old_net = net.toJSON();
        net = new convnetjs.Net();
        net.fromJSON(JSON.parse(data.net));
        reset_all();
        batch_num = data.batch_num;
    });
    return batch_num;
}
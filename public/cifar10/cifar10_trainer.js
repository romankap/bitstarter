var layer_defs, net, trainer;
var old_net, curr_net, gradients_net;
var samples_in_batch = 1000;
var curr_model_ID = 0, curr_sample_num=-1;
var is_net_loaded_from_server = false, is_training_active = false;
var train_on_batch_interval;

var fw_timings_sum, fw_timings_num;
var bw_timings_sum, bw_timings_num;
var latency_to_server, post_to_server_start, post_to_server_end;
var latency_from_server, get_from_server_start, get_from_server_end;

var gradients_calculator = {
    traverse: function (new_net_property, old_net_property, prev_property) {
        debug_global_counter = 0;

        for (var i in new_net_property) {
            if (new_net_property[i] !== null && typeof(new_net_property[i]) == "object") {
                //going on step down in the object tree!!
                this.traverse(new_net_property[i], old_net_property[i], i);
            }
            else if (new_net_property[i] !== null && typeof(new_net_property[i]) !== "object" &&
                    old_net_property[i] !== null && typeof(old_net_property[i]) !== "object" &&
                    new_net_property[i] !== NaN && old_net_property[i] !== NaN &&
                    prev_property == 'w' && isNumeric(i)) {
                new_net_property[i] -= old_net_property[i];
            }
        }
    }
}

var reset_stats = function() {
    fw_timings_sum = fw_timings_num = 0;
    bw_timings_sum = bw_timings_num = 0;
    post_to_server_start = post_to_server_end = -1;
    get_from_server_start = get_from_server_end = -1;
}

var add_to_fw_and_bw_timing_stats = function(fw_time, bw_time) {
    fw_timings_sum += fw_time;
    fw_timings_num++;

    bw_timings_sum += bw_time;
    bw_timings_num++;
}

var get_fw_timings_average = function () {
    if (fw_timings_num > 0)
        return fw_timings_sum / fw_timings_num;
    return null;
}

var get_bw_timings_average = function () {
    if (bw_timings_num > 0)
        return bw_timings_sum / bw_timings_num;
    return null;
}
// ------------------------
// BEGIN CIFAR10 SPECIFIC STUFF
// ------------------------
var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
var use_validation_data = true;

function isNumeric(num) {
    return !isNaN(num)
}

// int main
$(window).load(function() {
    client_ID = make_string_ID();
    change_client_name();
    console.log("Hello, I am trainer-client " + client_ID);
});

// loads a training image and trains on it with the network
var load_and_step_interval;

var train_on_batch = function() {
    if (is_training_active) {
        var sample = sample_training_instance(curr_sample_num);
        step(sample, curr_sample_num); // process this image
        curr_sample_num++
        if(curr_sample_num === 1000) {
            post_gradients_to_server();
            clearInterval(train_on_batch_interval);
        }
    }
}

var start_working = function() {
    if (is_net_loaded_from_server && is_batch_loaded) {
        is_training_active = true;
        curr_sample_num = 0;
        train_on_batch_interval = setInterval(train_on_batch, 0);
        console.log('Starting to work!');
    }
    else {
        setTimeout(start_working, 200);
    }
}

var paused = true;

var compute = function() {
    paused = !paused;
    var btn = document.getElementById('compute-btn');
    if (paused) {
        btn.value = 'compute';
        is_training_active = false;
        clearInterval(train_on_batch_interval);
        post_gradients_to_server();
    }
    else {
        btn.value = 'pause';
        get_net_and_update_batch_from_server(); //calls start_working();
    }
}

var sample_training_instance = function (sample_num) {

    // fetch the appropriate row of the training image and reshape into a Vol
    var p = img_data.data;
    var new_Vol = new convnetjs.Vol(32,32,3,0.0);
    var W = 32*32;
    var j=0;
    for(var dc=0;dc<3;dc++) {
        var i=0;
        for(var xc=0;xc<32;xc++) {
            for(var yc=0;yc<32;yc++) {
                var ix = ((W * sample_num) + i) * 4 + dc;
                new_Vol.set(yc,xc,dc,p[ix]/255.0-0.5);
                i++;
            }
        }
    }
    var dx = Math.floor(Math.random()*5-2);
    var dy = Math.floor(Math.random()*5-2);
    new_Vol = convnetjs.augment(new_Vol, 32, dx, dy, Math.random()<0.5); //maybe flip horizontally

    //var isval = use_validation_data && sample_num%10===0 ? true : false;
    var label_num = curr_batch_num * samples_in_batch + sample_num;
    return {x:new_Vol, label:labels[label_num]};
}

var step = function(sample, sample_num) {
    var x = sample.x;
    var y = sample.label;

    /*if(sample.isval) {
        // use x to build our estimate of validation error
        net.forward(x);
        var yhat = net.getPrediction();
        var val_acc = yhat === y ? 1.0 : 0.0;
        valAccWindow.add(val_acc);
        return; // get out
    }*/

    // train on it with network
    var stats = trainer.train(x, y);
    var lossx = stats.cost_loss;
    var lossw = stats.l2_decay_loss;
    add_to_fw_and_bw_timing_stats(stats.fwd_time, stats.bwd_time);

    // keep track of stats such as the average training error and loss
    var yhat = net.getPrediction();
    var train_acc = yhat === y ? 1.0 : 0.0;
    xLossWindow.add(lossx);
    wLossWindow.add(lossw);
    trainAccWindow.add(train_acc);

    // visualize training status
    var train_elt = document.getElementById("trainstats");
    train_elt.innerHTML = '';
    var t = 'Average forward time per example: ' + get_fw_timings_average().toFixed(2) + 'ms';
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Average backprop time per example: ' + get_bw_timings_average().toFixed(2) + 'ms';
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
    //var t = 'Validation accuracy: ' + f2t(valAccWindow.get_average());
    //train_elt.appendChild(document.createTextNode(t));
    //train_elt.appendChild(document.createElement('br'));
    var t = 'Examples seen (out of '+ samples_in_batch + "): " + sample_num;
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
}

var calculate_gradients = function() {
    gradients_calculator.traverse(gradients_net, old_net, "");
}

var post_gradients_to_server = function() {
    is_training_active = false;
    post_to_server_start = new Date().getTime();

    curr_net = net.toJSON();
    gradients_net = curr_net;
    calculate_gradients();
    var gradients_net_in_JSON_string = JSON.stringify(gradients_net);
    var learning_rate = trainer.learning_rate;
    var momentum = trainer.momentum;
    var l2_decay = trainer.l2_decay;

    //Also sending previous latency to server
    var parameters = {model_name: "CIFAR10", net: gradients_net_in_JSON_string, model_ID: curr_model_ID,
                    client_ID: client_ID ,learning_rate :learning_rate, momentum: momentum, l2_decay: l2_decay,
                    fw_timings_average: get_fw_timings_average().toFixed(2), bw_timings_average: get_bw_timings_average().toFixed(2),
                    latency_to_server: latency_to_server};
    console.log("Sending CIFAR10 net_in_JSON with length " + parameters.net.length);
    //console.log("Sending CIFAR10 net: " + parameters.net.substring(0,1000));
    $.post('/update_model_from_gradients', parameters, function(data) {
        console.log(data);
        post_to_server_end = new Date().getTime();
        latency_to_server = post_to_server_end - post_to_server_start;
        console.log("<post_gradients_to_server> Sending latency_to_server: " + latency_to_server);
    });

    if (!paused)
        get_net_and_update_batch_from_server();
}

var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        return data.batch_num;
    });
}

var get_net_and_update_batch_from_server = function() {
    get_from_server_start = new Date().getTime();

    //Sending previous latency from server
    var parameters = {model_name: "CIFAR10", client_ID: client_ID, latency_from_server: latency_from_server};
    console.log("<get_net_and_update_batch_from_server> Sending latency_from_server: " + latency_from_server);
    is_net_loaded_from_server = false;

    $.get('/get_net_and_update_batch_from_server', parameters, function(data) {
        curr_batch_num = data.batch_num;
        curr_epoch_num = data.epoch_num;
        load_data_batch(curr_batch_num);

        update_displayed_batch_and_epoch_nums(curr_batch_num, curr_epoch_num);

        //old_net.fromJSON(net.toJSON());
        curr_net = JSON.parse(data.net);
        old_net = curr_net;
        net = new convnetjs.Net();
        net.fromJSON(curr_net);
        trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});

        trainer.learning_rate = data.learning_rate;
        trainer.momentum = data.momentum;
        trainer.l2_decay = data.l2_decay;
        curr_model_ID  = data.model_ID;

        reset_all();

        is_net_loaded_from_server = true;

        get_from_server_end = new Date().getTime();
        latency_from_server = get_from_server_end - get_from_server_start;

        reset_stats();
        console.log("<get_net_and_update_batch_from_server> Received " + parameters.model_name + " net back. model_ID: " + data.model_ID);
        console.log("<get_net_and_update_batch_from_server> Working on batch: " + data.batch_num); //DEBUG
        console.log("<get_net_and_update_batch_from_server> Received " + data.net.length + " net in length back"); //DEBUG
    });
    start_working();
}
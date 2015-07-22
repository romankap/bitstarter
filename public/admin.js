var layer_defs, net, trainer;
var curr_model_ID=-1, curr_epoch_num=0;
var curr_sample_num=0;

var total_samples_predicted=0, total_predicted_correctly=0;
var curr_net_accuracy=0, curr_validation_accuracy=0;

// ------------------------
// BEGIN CIFAR10 SPECIFIC STUFF
// ------------------------
var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

var use_validation_data = true;
var first_execution = true;
var check_net_accuracy_frequency = 20 * 1000;
var total_training_batches; //TODO: change batch numbers and inputs accordingly
var samples_in_test_batch, samples_in_validation_batch;
var get_validation_model_interval, validation_batch_interval;
var wait_for_testing_net_to_load_interval, testing_batch_interval;
var get_net_accuracy = false, minimum_epochs_to_train;
var is_net_loaded_from_server = false;
var is_admin_in_testing_mode = false;

var initialize_model_parameters = function(data) {
    total_training_batches = data.total_training_batches;
    samples_in_test_batch = data.samples_in_testing_batch;
    samples_in_validation_batch = data.samples_in_validation_batch;
    minimum_epochs_to_train = data.minimum_epochs_to_train;
}

// int main
$(window).load(function() {
    console.log("Hello Admin, your wish is net's command");
    load_data_batch("validation");
    var AJAX_init_parameters = {model_name: "CIFAR10" };
    $.get('/get_init_model_from_server', AJAX_init_parameters, function(data) {
        console.log("Received an init_model from server: \n" + data.init_model);

        initialize_model_parameters(data);
        init_model = data.init_model;
        $("#newnet").val(init_model);
        eval(init_model);
        //net.fromJSON(JSON.parse(data.net));
        reset_all();
        update_net_param_display();

        is_batch_loaded = false;

        //load_data_batch(0); // async load train set batch 0 (6 total train batches)
        //load_data_batch(validate_batch); // async load test set (batch 6)
        //start_fun();
    });
});

var get_testing_model_from_server = function () {
    is_net_loaded_from_server = false;
    get_net_and_current_training_batch_from_server();
}


var sample_test_instance = function(sample_num_to_test) {
    return sample_image_instance(label_num_in_test_batch, sample_num_to_test);
}


var test_batch = function() {
    if (!get_net_accuracy) return;

    if(total_samples_predicted < samples_in_test_batch)
        predict_samples_group(sample_test_instance);
    else{
        $("#total-samples-tested").text("From " + samples_in_test_batch + "samples in the entire TEST-set");

        clearInterval(testing_batch_interval);
        clearInterval(wait_for_testing_net_to_load_interval);
        is_net_loaded_from_server = false;
        toggle_validate();
    }
    //var vis_elt = document.getElementById("visnet");
    //visualize_activations(net, vis_elt);
    //update_net_param_display();
}

var wait_for_testing_net_to_load = function() {
    if (is_net_loaded_from_server) {
        console.log('Starting TO TEST');
        clearInterval(wait_for_testing_net_to_load_interval);
        testing_batch_interval = setInterval(test_batch, 0);
    }
    else {
        wait_for_testing_net_to_load_interval = setTimeout(wait_for_testing_net_to_load, 200);
    }
}

var get_testing_accuracy = function(){
    clearInterval(get_validation_model_interval);
    clearInterval(validation_batch_interval);

    get_testing_model_from_server();
    wait_for_testing_net_to_load();
}

var get_validation_model_from_server = function () {
    if(get_net_accuracy) {
        get_net_and_current_training_batch_from_server();
        validation_batch_interval = setInterval(validate_batch, 0);
    }
    else
        clearInterval(get_validation_model_interval);
}

var validate_batch = function() {
    if (!get_net_accuracy || !is_net_loaded_from_server) return;

    if(total_samples_predicted < samples_in_validation_batch){
        predict_samples_group(sample_validation_instance);
    }
    else{
        $("#total-samples-tested").text("From " + samples_in_validation_batch + " validation-set samples. Plotted weight activations");
        var vis_elt = document.getElementById("visnet");
        visualize_activations(net, vis_elt);

        is_net_loaded_from_server = false;
        store_validation_accuracy_on_server();
        clearInterval(validation_batch_interval);
        get_validation_model_interval = setInterval(get_validation_model_from_server, check_net_accuracy_frequency);
    }
    //var vis_elt = document.getElementById("visnet");
    //visualize_activations(net, vis_elt);
    //update_net_param_display();
}

var start_validating = function() {
    if (is_batch_loaded) {
        console.log('Starting validation');
        validation_batch_interval = setInterval(validate_batch, 0);
    }
    else {
        setTimeout(start_validating, 200);
    }
}

var toggle_validate = function () {
    var btn = document.getElementById('toggle-validate-btn');
    get_net_accuracy = !get_net_accuracy;

    if (get_net_accuracy) {
        if ($('#restart-checkbox').is(":checked"))
            curr_epoch_num = -1;
        get_validation_model_interval = setInterval(get_validation_model_from_server, check_net_accuracy_frequency);
        get_validation_model_from_server();

        btn.innerHTML = '<i class="fa fa-stop"></i> Stop Validating'
        start_validating();
    }
    else {
        clearInterval(get_validation_model_interval);
        clearInterval(validation_batch_interval);
        btn.innerHTML = '<i class="fa fa-play-circle"></i> Start Validating'
    }
}

////////////////////////////////////////////

var update_contributing_clients = function(total_different_clients, last_contributing_client) {
    if (total_different_clients != undefined && last_contributing_client != undefined && total_different_clients > 0) {
        $('#total-clients').html("total different clients: " + total_different_clients +
                                " , last contributing client: " + last_contributing_client);
    }
}

var label_num_in_validation_batch = function(sample_num_to_validate) {
    //return total_training_batches*samples_in_test_batch + sample_num_to_validate; //TODO: replace this line with the following
    return sample_num_to_validate;
}


var label_num_in_test_batch = function(sample_num_to_test) {
    //return total_training_batches*samples_in_test_batch + sample_num_to_test; //OLD
    return samples_in_validation_batch + sample_num_to_test; //TODO: replace this line with the following one
    //return samples_in_validation_batch + sample_num_to_test; //test samples follow the validation samples
}


// sample a random testing instance
var sample_image_instance = function(get_label_num ,sample_num_to_predict) {
    if (sample_num_to_predict === undefined)
        var sample_num_to_predict = get_random_number(samples_in_test_batch);

    var p = img_data.data;
    var x = new convnetjs.Vol(32,32,3,0.0);
    var W = 32*32;
    var j=0;
    for(var dc=0;dc<3;dc++) {
        var i=0;
        for(var xc=0;xc<32;xc++) {
            for(var yc=0;yc<32;yc++) {
                var ix = ((W * sample_num_to_predict) + i) * 4 + dc;
                x.set(yc,xc,dc,p[ix]/255.0-0.5);
                i++;
            }
        }
    }

    var label_index = get_label_num(sample_num_to_predict);
    return {x:x, label:labels[label_index]};
}

var sample_validation_instance = function(sample_num_to_test) {
    return sample_image_instance(label_num_in_validation_batch, sample_num_to_test);
}

// Goes over the entire testing batch and updates curr_validation_accuracy
var get_validation_score = function() {

}


// Goes over the entire testing batch and updates curr_net_accuracy
var predict_samples_group = function(sample_instance_function) {
    var num_classes = net.layers[net.layers.length-1].out_depth;

    for(var num=0;num<20;num++) {
        var sample = sample_instance_function(curr_sample_num);
        var sample_label = sample.label;  // ground truth label

        // forward prop it through the network
        var aavg = new convnetjs.Vol(1,1,num_classes,0.0);
        // ensures we always have a list, regardless if above returns single item or list
        var xs = [].concat(sample.x);
        var n = xs.length;
        for(var i=0;i<n;i++) {
            var a = net.forward(xs[i]);
            aavg.addFrom(a);
        }
        var preds = [];
        for(var k=0;k<aavg.w.length;k++) { preds.push({k:k,p:aavg.w[k]}); }
        preds.sort(function(a,b){return a.p<b.p ? 1:-1;});

        var correct = preds[0].k===sample_label;
        if(correct) total_predicted_correctly++;
        total_samples_predicted++;

        curr_sample_num++;
    }
    curr_net_accuracy = total_predicted_correctly / total_samples_predicted;
    if(!is_admin_in_testing_mode) {
        $("#testset_acc").text('average validation accuracy: ' + curr_net_accuracy.toFixed(2));
        $("#total-samples-tested").text('Based on ' + total_samples_predicted + " samples");
    }
    else{
        $("#testset_acc").text('average TESTING accuracy: ' + curr_net_accuracy.toFixed(2));
        $("#total-samples-tested").text('Based on ' + total_samples_predicted + " samples");
    }
    //console.log("<predict_samples_group> finished predicting total_samples_predicted: " + total_samples_predicted);
}


var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        curr_batch_num = data.batch_num - 1;
        return curr_batch_num;
    });
}

var load_net_from_server_data = function(data_from_server) {
    var net_in_JSON = JSON.parse(data_from_server.net);
    net = new convnetjs.Net();
    net.fromJSON(net_in_JSON);
    trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});

    trainer.learning_rate = data_from_server.learning_rate;
    trainer.momentum = data_from_server.momentum;
    trainer.l2_decay = data_from_server.l2_decay;
    curr_model_ID  = data_from_server.model_ID;
    curr_epoch_num = data_from_server.epoch_num;

    reset_all();
    curr_sample_num=0;
    total_samples_predicted=0;
    total_predicted_correctly=0;
}

var get_net_and_current_training_batch_from_server = function() {
    var parameters = {model_name: "CIFAR10", model_ID: curr_model_ID, epoch_num: curr_epoch_num };
    $.get('/get_net_and_current_training_batch_from_server', parameters, function(data) {
        if (data.batch_num == 0)
            curr_batch_num = 0;
        else
            curr_batch_num = (data.batch_num-1) % total_training_batches;

        console.log("<get_net_and_current_training_batch_from_server> Received "+ parameters.model_name + " model with model_ID: " +
                    data.model_ID + " and epoch_num " + data.epoch_num);

        if (data.model_ID !== curr_model_ID || data.epoch_num !== curr_epoch_num || is_admin_in_testing_mode){
            load_net_from_server_data(data); //Load a new net for validation / testing

            is_net_loaded_from_server = true;
            console.log("<get_net_and_current_training_batch_from_server> model_ID & epoch_num were updated ==> LOADING new net");
        }
        else
            console.log("<get_net_and_current_training_batch_from_server> model_ID & epoch_num DIDN'T update ==> KEEPING old net");

        update_displayed_batch_and_epoch_nums(curr_batch_num, curr_epoch_num);
        update_contributing_clients(data.total_different_clients, data.last_contributing_client);

        //var vis_elt = document.getElementById("visnet");

        //test_predict();
        //visualize_activations(net, vis_elt);
        //update_net_param_display();
    });
}

var get_validation_model = function() {
    var parameters = {model_name: "CIFAR10", model_ID: curr_model_ID};
    $.get('/get_validation_net', parameters, function(data) {
        load_net_from_server_data(data);

        is_net_loaded_from_server = true;
    });
}

var store_validation_accuracy_on_server = function() {
    var parameters = {model_ID: curr_model_ID, epoch_num: curr_epoch_num,
                    validation_accuracy: curr_net_accuracy};
    $.post('/store_validation_accuracy_on_server', parameters, function(data) {
        console.log("<store_validation_accuracy_on_server> Received data.is_testing_needed: " + data.is_testing_needed);
        if(data.is_testing_needed) {
            console.log("<store_validation_accuracy_on_server> Worse accuracy after " + curr_epoch_num +
                        " epochs, GOING TO TESTING MODE");
            is_admin_in_testing_mode = true;
            get_testing_accuracy(); //TODO: remove comment from this line and implement function
        }
        else
            console.log("<store_validation_accuracy_on_server> Stored testing accuracy " + curr_net_accuracy);
    });
}

/*
var get_all_stats = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_all_stats', parameters, function(data) {
        var stats_received = data;//.toString();//.replace('<newline>' ,/\n/);
        console.log("<get_all_stats> Received the following stats: " + stats_received);
        $('#download-stats-csv').attr("href", "data:text/plain;charset=utf-8," + stats_received);
        $('#download-stats-csv').attr("download", "stats.csv");
        $('#download-stats-csv').show();
    });
}*/

var reset_model = function() {
    var parameters = {model_name: "CIFAR10"};

    $.post('/reset_model', parameters, function(data) {
        console.log("Resetting the model named: <" + parameters.model_name + "> stored on server");
    });

    is_net_loaded_from_server = false;
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
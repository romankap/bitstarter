var net, trainer;

var dataset_name;

var model_ID;

var layer_defs, net, trainer;
var curr_model_ID=-1, curr_epoch_num=0;
var curr_sample_num=0;

var total_samples_predicted=0, total_predicted_correctly=0;
var curr_net_accuracy=0;

var data_img_elt;
var img_data;

var model_id;


var check_net_accuracy_frequency = 60 * 1000;
var total_training_batches;
var samples_in_test_batch, samples_in_validation_batch;
var get_validation_model_interval, validation_batch_interval;
var wait_for_testing_net_to_load_interval, testing_batch_interval;
var get_net_accuracy = false, minimum_epochs_to_train;
var is_net_loaded_from_server = false;
var is_admin_in_testing_mode = false;

var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var testAccWindow = new cnnutil.Window(50, 1);

var curr_batch_num;

var update_net_param_display = function() {
    document.getElementById('lr_input').value = trainer.learning_rate;
    document.getElementById('momentum_input').value = trainer.momentum;
    document.getElementById('batch_size_input').value = trainer.batch_size;
    document.getElementById('decay_input').value = trainer.l2_decay;
}


var initialize_model_parameters = function(data) {
    total_training_batches = data.total_training_batches;
    samples_in_test_batch = data.samples_in_testing_batch;
    samples_in_validation_batch = data.samples_in_validation_batch;
    minimum_epochs_to_train = data.minimum_epochs_to_train;
}

var load_data_batch = function(batch_url) {
    // Load the dataset with JS in background
    if(is_batch_loaded) return;

    data_img_elt = new Image();
    data_img_elt.crossOrigin = 'anonymous';

    data_img_elt.onload = function() {
        var data_canvas = document.createElement('canvas');
        data_canvas.width = data_img_elt.width;
        data_canvas.height = data_img_elt.height;
        var data_ctx = data_canvas.getContext("2d");
        data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
        img_data = data_ctx.getImageData(0, 0, data_img_elt.width, data_img_elt.height);
        is_batch_loaded = true;
        console.log('Finished loading data batch #' + batch_url);
    }

    data_img_elt.src = batch_url;
}

// int main
$(window).load(function() {
    console.log("Hello Admin, your wish is net's command");

     reset_all();
    $.get('/get_base_model_from_server', function(data) {
      //  console.log("Received an init_model from server: \n" + data.init_model);

        $.get('/get_admin_batch', function(data) {
            load_data_batch(data);
        });

        net = new convnetjs.Net();
        net.fromJSON(data.base_net);
        if(data.model_ID != model_id) {
          $.getScript(data.dataset + '/labels_verify.js');
          dataset_name = data.dataset;
          $("#data_name").text(data.dataset.toUpperCase());
        }
        trainer = new convnetjs.SGDTrainer(net, data.trainer_param);

        model_id  = data.model_ID;

        initialize_model_parameters(data);

        is_net_loaded_from_server = true;


        update_net_param_display();
        load_net_schem();
    });
});

var reset_all = function() {    // Just reset any visual cues of doing anything
    is_net_loaded_from_server = false;
    is_batch_loaded = false;
    is_training_active = false;
    curr_sample_num = 0;

    // reinit windows that keep track of val/train accuracies
    xLossWindow.reset();
    wLossWindow.reset();
  //  trainAccWindow.reset();
    valAccWindow.reset();
    testAccWindow.reset();
    lossGraph = new cnnvis.Graph(); // reinit graph too
}

var get_testing_model_from_server = function () {
    is_net_loaded_from_server = false;
    get_net_and_current_training_batch_from_server();
}


var sample_test_instance = function(sample_num_to_test) {
    return sample_image_instance[dataset_name](label_num_in_test_batch, sample_num_to_test);
}


var test_batch = function() {
    if (!get_net_accuracy) return;

    if(total_samples_predicted < samples_in_test_batch) {
        predict_samples_group(sample_test_instance);
        setTimeout(test_batch, 0);
    }
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
        testing_batch_interval = setTimeout(test_batch, 0);
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

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {
    var s = scale || 2; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    // get max and min activation to scale the maps automatically
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    // create the canvas elements, draw and add to DOM
    for(var d=0;d<A.depth;d++) {

        var canv = document.createElement('canvas');
        canv.className = 'actmap';
        var W = A.sx * s;
        var H = A.sy * s;
        canv.width = W;
        canv.height = H;
        var ctx = canv.getContext('2d');
        var g = ctx.createImageData(W, H);

        for(var x=0;x<A.sx;x++) {
            for(var y=0;y<A.sy;y++) {
                if(draw_grads) {
                    var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
                } else {
                    var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);
                }
                for(var dx=0;dx<s;dx++) {
                    for(var dy=0;dy<s;dy++) {
                        var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                        for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
                        g.data[pp+3] = 255; // alpha channel
                    }
                }
            }
        }
        ctx.putImageData(g, 0, 0);
        elt.appendChild(canv);
    }
}

var draw_activations_COLOR = function(elt, A, scale, grads) {

    var s = scale || 2; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    // get max and min activation to scale the maps automatically
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);
    for(var d=0;d<3;d++) {
        for(var x=0;x<A.sx;x++) {
            for(var y=0;y<A.sy;y++) {
                if(draw_grads) {
                    var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
                } else {
                    var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);
                }
                for(var dx=0;dx<s;dx++) {
                    for(var dy=0;dy<s;dy++) {
                        var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                        g.data[pp + d] = dval;
                        if(d===0) g.data[pp+3] = 255; // alpha channel
                    }
                }
            }
        }
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
}

var visualize_activations = function(net, elt) {

    // clear the element
    elt.innerHTML = "";

    // show activations in each layer
    var N = net.layers.length;
    for(var i=0;i<N;i++) {
        var L = net.layers[i];

        var layer_div = document.createElement('div');

        // visualize activations
        var activations_div = document.createElement('div');
        activations_div.appendChild(document.createTextNode('Activations:'));
        activations_div.appendChild(document.createElement('br'));
        activations_div.className = 'layer_act';
        var scale = 2;
        if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax

        // HACK to draw in color in input layer
        if(i===0) {
            draw_activations_COLOR(activations_div, L.out_act, scale);
            draw_activations_COLOR(activations_div, L.out_act, scale, true);

            /*
             // visualize positive and negative components of the gradient separately
             var dd = L.out_act.clone();
             var ni = L.out_act.w.length;
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq > 0 ? dwq : 0.0; }
             draw_activations_COLOR(activations_div, dd, scale);
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq < 0 ? -dwq : 0.0; }
             draw_activations_COLOR(activations_div, dd, scale);
             */

            /*
             // visualize what the network would like the image to look like more
             var dd = L.out_act.clone();
             var ni = L.out_act.w.length;
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] -= 20*dwq; }
             draw_activations_COLOR(activations_div, dd, scale);
             */

            /*
             // visualize gradient magnitude
             var dd = L.out_act.clone();
             var ni = L.out_act.w.length;
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq*dwq; }
             draw_activations_COLOR(activations_div, dd, scale);
             */

        } else {
            draw_activations(activations_div, L.out_act, scale);
        }

        // visualize data gradients
        if(L.layer_type !== 'softmax' && L.layer_type !== 'input' ) {
            var grad_div = document.createElement('div');
            grad_div.appendChild(document.createTextNode('Activation Gradients:'));
            grad_div.appendChild(document.createElement('br'));
            grad_div.className = 'layer_grad';
            var scale = 2;
            if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax
            draw_activations(grad_div, L.out_act, scale, true);
            activations_div.appendChild(grad_div);
        }

        // visualize filters if they are of reasonable size
        if(L.layer_type === 'conv') {
            var filters_div = document.createElement('div');
            if(L.filters[0].sx>3) {
                // actual weights
                filters_div.appendChild(document.createTextNode('Weights:'));
                filters_div.appendChild(document.createElement('br'));
                for(var j=0;j<L.filters.length;j++) {
                    // HACK to draw in color for first layer conv filters
                    if(i===1) {
                        draw_activations_COLOR(filters_div, L.filters[j], 2);
                    } else {
                        filters_div.appendChild(document.createTextNode('('));
                        draw_activations(filters_div, L.filters[j], 2);
                        filters_div.appendChild(document.createTextNode(')'));
                    }
                }
                // gradients
                filters_div.appendChild(document.createElement('br'));
                filters_div.appendChild(document.createTextNode('Weight Gradients:'));
                filters_div.appendChild(document.createElement('br'));
                for(var j=0;j<L.filters.length;j++) {
                    if(i===1) { draw_activations_COLOR(filters_div, L.filters[j], 2, true); }
                    else {
                        filters_div.appendChild(document.createTextNode('('));
                        draw_activations(filters_div, L.filters[j], 2, true);
                        filters_div.appendChild(document.createTextNode(')'));
                    }
                }
            } else {
                filters_div.appendChild(document.createTextNode('Weights hidden, too small'));
            }
            activations_div.appendChild(filters_div);
        }
        layer_div.appendChild(activations_div);

        // print some stats on left of the layer
        layer_div.className = 'layer ' + 'lt' + L.layer_type;
        var title_div = document.createElement('div');
        title_div.className = 'ltitle'
        var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
        title_div.appendChild(document.createTextNode(t));
        layer_div.appendChild(title_div);

        if(L.layer_type==='conv') {
            var t = 'filter size ' + L.filters[0].sx + 'x' + L.filters[0].sy + 'x' + L.filters[0].depth + ', stride ' + L.stride;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }
        if(L.layer_type==='pool') {
            var t = 'pooling size ' + L.sx + 'x' + L.sy + ', stride ' + L.stride;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }

        // find min, max activations and display them
        var mma = maxmin(L.out_act.w);
        var t = 'max activation: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
        layer_div.appendChild(document.createTextNode(t));
        layer_div.appendChild(document.createElement('br'));

        var mma = maxmin(L.out_act.dw);
        var t = 'max gradient: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
        layer_div.appendChild(document.createTextNode(t));
        layer_div.appendChild(document.createElement('br'));

        // number of parameters
        if(L.layer_type==='conv' || L.layer_type==='local') {
            var tot_params = L.sx*L.sy*L.in_depth*L.filters.length + L.filters.length;
            var t = 'parameters: ' + L.filters.length + 'x' + L.sx + 'x' + L.sy + 'x' + L.in_depth + '+' + L.filters.length + ' = ' + tot_params;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }
        if(L.layer_type==='fc') {
            var tot_params = L.num_inputs*L.filters.length + L.filters.length;
            var t = 'parameters: ' + L.filters.length + 'x' + L.num_inputs + '+' + L.filters.length + ' = ' + tot_params;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }

        // css madness needed here...
        var clear = document.createElement('div');
        clear.className = 'clear';
        layer_div.appendChild(clear);

        elt.appendChild(layer_div);
    }
}

var validate_batch = function() {
    if (!get_net_accuracy || !is_net_loaded_from_server || !is_batch_loaded) return;

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

var get_validation_model_from_server = function () {
    if(get_net_accuracy ) {
        get_net_and_current_training_batch_from_server();
        validation_batch_interval = setInterval(validate_batch, 1);
    }
    else
        clearInterval(get_validation_model_interval);
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
var sample_image_instance = {
    cifar10: function(get_label_num ,sample_num_to_predict) {
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
  },
  mnist: function (get_label_num ,sample_num_to_predict) {
       // fetch the appropriate row of the training image and reshape into a Vol
      var p = img_data.data;
      var x = new convnetjs.Vol(28,28,1,0.0);
      var W = 28*28;
      for(var i=0;i<W;i++) {
        var ix = ((W * sample_num_to_predict) + i) * 4;
        x.w[i] = p[ix]/255.0;
      }
      x = convnetjs.augment(x, 24);

      var label_num = get_label_num(sample_num_to_predict);
      return {x:x, label:labels[label_num]};
  }

};

var sample_validation_instance = function(sample_num_to_test) {
    return sample_image_instance[dataset_name](label_num_in_validation_batch, sample_num_to_test);
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

        net.forward(sample.x);
        var yhat = net.getPrediction();
        if( yhat === sample_label) {
            total_predicted_correctly++;
        }
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

    $.get('/get_batch_num_from_server', function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        curr_batch_num = data.batch_num;
        return curr_batch_num;
    });
}

var load_net_from_server_data = function(data_from_server) {

    net = new convnetjs.Net();
    net.fromJSON(data_from_server.net);
    trainer = new convnetjs.SGDTrainer(net, data_from_server.model_param);

    curr_model_ID  = data_from_server.model_ID;
    curr_epoch_num = data_from_server.epoch_num;

    reset_all();
    curr_sample_num=0;
    total_samples_predicted=0;
    total_predicted_correctly=0;
}

var update_displayed_batch_and_epoch_nums = function(new_batch_num, new_epoch_num, num_of_different_clients) {
    if (new_batch_num === -1)
        $('#batch-num').html("Waiting for batch and epoch to update");
    else {
        $('#batch-num').html("Training on batch #" + new_batch_num + " , epoch #" + new_epoch_num);
        if (num_of_different_clients !== undefined && num_of_different_clients >= 2) {
            $('#clients-connected').html('You are currently part of a net with ' +
                '<span class=\"color-blue\">' + num_of_different_clients + "</span> participants, thank you!");
        }
        else if (num_of_different_clients !== undefined ) {
            $('#clients-connected').html('You are the first trainer, way to go!');
        }
    }
}

var get_net_and_current_training_batch_from_server = function() {
    var parameters = { model_ID: curr_model_ID };
    if(curr_batch_num == undefined) return

    $.get('/get_net_and_current_training_batch_from_server', parameters, function(data) {
        curr_batch_num = data.batch_num;


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
    var parameters = { model_ID: curr_model_ID };
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

var change_dataset = function(dataset) {
  $.post('/change_dataset', {name: dataset}, function(data) {
      location.reload();
  });
}

var load_net_schem = function() {
  $.get('/get_current_net_schem', function(data) {
      $("#newnet").text(data);
  });
}

var change_net = function() {
    var new_init_model = $("#newnet").val();
    //rand = get_random_number();
    var parameters = {new_init_model: new_init_model};
    console.log("<change_net> Sending the following new CIFAR10 init_model: " + new_init_model);
    reset_all();
    $.post('/store_new_model_on_server', parameters, function(data) {
        console.log(data);
    });
}

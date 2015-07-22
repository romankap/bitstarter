var layer_defs, net, trainer;
var curr_model_ID;
var curr_sample_num;


var use_validation_data = true;
var first_execution = true;
var validation_frequency = 30 * 1000;
var validation_batch_num = 50;
var samples_in_test_batch = 1000;
var validation_interval, validate_batch_interval;
var get_validations = false, is_net_loaded_from_server = false;

var dataset_id;


// int main
$(window).load(function() {
    // load_data_batch(validation_batch_num);
    $.get('/get_base_model_from_server', function(data) {
        console.log("Received an base_model from server: \n" + data.init_model);

        init_model = data.init_model;
        $("#newnet").val(init_model);
        eval(init_model);
        //net.fromJSON(JSON.parse(data.net));
        reset_all();
        update_net_param_display();

        is_batch_loaded = false;
    });
});

var test_prediction_accuracy = function () {
    get_net_and_batch_from_server();
}

var validate_batch = function() {
    if (!get_validations) return;

    //curr_sample_num = get_random_number(samples_in_test_batch);
    //var sample = sample_test_instance(curr_sample_num);
    //step(sample, curr_sample_num); // process this image

    test_predict();
    var vis_elt = document.getElementById("visnet");
    visualize_activations(net, vis_elt);
    update_net_param_display();
}

var start_validating = function() {
    if (is_batch_loaded) {
        console.log('Starting validation');
        validate_batch_interval = setInterval(validate_batch, 0);
    }
    else {
        setTimeout(start_validating, 200);
    }
}

var toggle_validate = function () {
    var btn = document.getElementById('toggle-validate-btn');
    get_validations = !get_validations;

    if (get_validations) {
        validation_interval = setInterval(test_prediction_accuracy, validation_frequency);
        btn.value = 'Stop Validating';
        start_validating();
    }
    else {
        clearInterval(validation_interval);
        clearInterval(validate_batch_interval);
        btn.value = 'Start Validating';
    }
}


// sample a random testing instance
var sample_test_instance = function() {
    var random_num = get_random_number(samples_in_test_batch);

    var p = img_data.data;
    var x = new convnetjs.Vol(32,32,3,0.0);
    var W = 32*32;
    var j=0;
    for(var dc=0;dc<3;dc++) {
        var i=0;
        for(var xc=0;xc<32;xc++) {
            for(var yc=0;yc<32;yc++) {
                var ix = ((W * random_num) + i) * 4 + dc;
                x.set(yc,xc,dc,p[ix]/255.0-0.5);
                i++;
            }
        }
    }

    // distort position and maybe flip
    var distorted_sample = [];
    //distorted_sample.push(x, 32, 0, 0, false); // push an un-augmented copy
    for(var k=0;k<6;k++) {
        var dx = Math.floor(Math.random()*5-2);
        var dy = Math.floor(Math.random()*5-2);
        distorted_sample.push(convnetjs.augment(x, 32, dx, dy, k>2));
    }

    var label_index = validation_batch_num*samples_in_test_batch + random_num ;
    // return multiple augmentations, and we will average the network over them
    // to increase performance
    return {x:distorted_sample, label:labels[label_index]};
}

// evaluate current network on test set
var test_predict = function() {
    var num_classes = net.layers[net.layers.length-1].out_depth;

    document.getElementById('testset_acc').innerHTML = '';
    var num_total = 0;
    var num_correct = 0;

    // grab a random test image
    for(var num=0;num<4;num++) {
        var sample = sample_test_instance();
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
        if(correct) num_correct++;
        num_total++;

        var div = document.createElement('div');
        div.className = 'testdiv';

        // draw the image into a canvas
        draw_activations_COLOR(div, xs[0], 2); // draw Vol into canv

        // add predictions
        var probsdiv = document.createElement('div');
        div.className = 'probsdiv';
        var t = '';
        for(var k=0;k<3;k++) {
            var col = preds[k].k===sample_label ? 'rgb(85,187,85)' : 'rgb(187,85,85)';
            t += '<div class=\"pp\" style=\"width:' + Math.floor(preds[k].p/n*100) + 'px; margin-left: 70px; background-color:' + col + ';\">' + classes_txt[preds[k].k] + '</div>'
        }
        probsdiv.innerHTML = t;
        div.appendChild(probsdiv);

        // add it into DOM
        $(div).prependTo($("#testset_vis")).hide().fadeIn('slow').slideDown('slow');
        if($(".probsdiv").length>200) {
            $("#testset_vis > .probsdiv").last().remove(); // pop to keep upper bound of shown items
        }
    }
    testAccWindow.add(num_correct/num_total);
    $("#testset_acc").text('average validation accuracy: ' + testAccWindow.get_average().toFixed(2));
    //console.log("num_correct: " + num_correct + " | num_total: " + num_total);
    //console.log('test accuracy : ' + testAccWindow.get_average().toFixed(2));
}


var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        curr_batch_num = data.batch_num;
        return data.batch_num;
    });
}

var get_net_and_batch_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_net_and_batch_from_server', parameters, function(data) {
        curr_model_ID  = data.model_ID;
        console.log("<get_net_and_batch_from_server> Received "+ parameters.model_name + " model with model_ID: " + curr_model_ID);

        var net_in_JSON = JSON.parse(data.net);
        net = new convnetjs.Net();
        net.fromJSON(net_in_JSON);
        trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});

        trainer.learning_rate = data.learning_rate;
        trainer.momentum = data.momentum;
        trainer.l2_decay = data.l2_decay;
        curr_model_ID  = data.model_ID;

        reset_all();

        curr_batch_num = data.batch_num;
        update_displayed_batch_num(curr_batch_num);

        //var vis_elt = document.getElementById("visnet");

        //test_predict();
        //visualize_activations(net, vis_elt);
        //update_net_param_display();

        is_net_loaded_from_server = true;
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

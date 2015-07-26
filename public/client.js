var net, trainer;

var data_img_elt;
var img_data;

var UPDATE_INTERVAL_MS = 500;

var client_name;

var is_batch_loaded;
var curr_batch_num;

var model_id;

var paused = true;

var old_net;

var curr_sample_num;
var is_net_loaded_from_server = false, is_training_active = false;
var train_on_batch_interval;    // Interval identifier


var fw_timings_sum, fw_timings_num;
var bw_timings_sum, bw_timings_num;
var latency_to_server, post_to_server_start, post_to_server_end;
var latency_from_server, get_from_server_start, get_from_server_end;

var dataset_name;

var make_string_ID = function ()
{
    var text = "";
    var consonants = "AEIO";
    var vowels = "BDFGJKLMNPRSTVYZ";

    for( var i=0; i < 5; i++ ) {
        if (i % 2 == 0)
            text += vowels.charAt(Math.floor(Math.random() * vowels.length));
        else
            text += consonants.charAt(Math.floor(Math.random() * consonants.length));
    }

    return text;
}

var change_client_name = function() {
    $('#client-name').html('Hello trainer-client <span class=\"color-blue\">' + client_ID + "</span>");
    $('#client-name-explanation').html("(" + client_ID + " is your identifying name in the crowdcomputing network)");
}



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
    //post_to_server_start = post_to_server_end = -1;
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

function isNumeric(num) {
    return !isNaN(num)
}

// Returns a random number in the range: [0, max_num-1]
var get_random_number = function (max_num) {
    return Math.floor((Math.random() * max_num));
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

var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var testAccWindow = new cnnutil.Window(50, 1);

var update_net_param_display = function() {
  //  document.getElementById('lr_input').value = trainer.learning_rate;
  //  document.getElementById('momentum_input').value = trainer.momentum;
  //  document.getElementById('batch_size_input').value = trainer.batch_size;
  //  document.getElementById('decay_input').value = trainer.l2_decay;
}

var clear_graph = function() {
    lossGraph = new cnnvis.Graph(); // reinit graph too
}

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


var start_client = function() {
    $.get('/start_client', function(data) {
        console.log("<start_client> Got net and parameters, starting to work on batch_num: " + data.batch_num);
        curr_batch_num = data.batch_num;
        load_data_batch(curr_batch_num);
        update_displayed_batch_num(curr_batch_num);
    });
}

var labels_loaded = true;

var init_all = function() {
    var parameters = {  client_ID: client_ID };

    $.get('/get_net_batch_all', parameters, function(data) {
		curr_batch_num = data.batch_num;
        curr_epoch_num = data.epoch_num;
        dataset_name = data.dataset_name;

        curr_batch_num = data.batch_num;
        batch_size = data.batch_size;

        load_data_batch(data.batch_url);
		    update_displayed_batch_and_epoch_nums(curr_batch_num, curr_epoch_num, data.total_different_clients);


        if(net != undefined) {
          old_net = net;
        }
        net = new convnetjs.Net();
        net.fromJSON(data.net);
        if(model_id != data.model_ID) {
          labels_loaded = false;
          $.getScript(data.dataset_name + '/labels.js', function() {
              labels_loaded = true;
          });
        }
        $("#data_name").text(data.dataset_name.toUpperCase());
        trainer = new convnetjs.SGDTrainer(net, data.trainer_param);

        model_id  = data.model_ID;

        is_net_loaded_from_server = true;

        get_from_server_end = new Date().getTime();
        latency_from_server = get_from_server_end - get_from_server_start;
		    reset_stats();



        console.log("<init_all> Received " + parameters.model_name + " net back. model_ID: " + data.model_ID);
        console.log("<init_all> Working on batch: " + data.batch_num); //DEBUG
        console.log("<init_all> Received " + data.net.length + " net in length back"); //DEBUG
    });
}

var post_gradients_to_server = function() {
  is_training_active = false;
  post_to_server_start = new Date().getTime();

  var gradients = net;
  if(old_net != undefined) {
    gradients_calculator.traverse(gradients, old_net, "");
  }
  console.log(gradients);

  var parameters = {
                      net: JSON.stringify(gradients),
                      model_ID: model_id,
                      client_ID: client_ID,
					  fw_timings_average: get_fw_timings_average().toFixed(2),
					  bw_timings_average: get_bw_timings_average().toFixed(2),
					  latency_to_server: latency_to_server
                  };

  console.log("Sending net_in_JSON with length " + parameters.net.length);

  $.post('/update_model_from_gradients', parameters, function(data) {
		console.log(data);
		post_to_server_end = new Date().getTime();
		latency_to_server = post_to_server_end - post_to_server_start;
		console.log("<post_gradients_to_server> Sending latency_to_server: " + latency_to_server);
  });
}

var start_working = function() {
    if (is_net_loaded_from_server && is_batch_loaded && labels_loaded) {
        is_training_active = true;
        curr_sample_num = 0;
        train_on_batch_interval = setTimeout(train_on_batch, 0);
        console.log('Starting to work!');
    }
    else {
        setTimeout(start_working, 200);
          console.log('Waiting to start to working...');
    }
}

var train_on_batch = function() {
    if (is_training_active) {
        var sample = sample_training_instance[dataset_name](curr_sample_num);
        step(sample, curr_sample_num); // process this image
        curr_sample_num++
        setTimeout(train_on_batch, 0);
        if(curr_sample_num == batch_size) {
            post_gradients_to_server();
            reset_all();
            if(!paused) {
              init_all();
              setTimeout(start_working, 0);
            }
        }
    }
}

var sample_training_instance = {    // desperate times call for desperate measures
    cifar10: function (sample_num, dims) {
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

        var label_num = curr_batch_num * batch_size + sample_num;
        return {x:new_Vol, label:labels[label_num]};
    },
    mnist: function (sample_num) {
         // fetch the appropriate row of the training image and reshape into a Vol
        var p = img_data.data;
        var x = new convnetjs.Vol(28,28,1,0.0);
        var W = 28*28;
        for(var i=0;i<W;i++) {
          var ix = ((W * sample_num) + i) * 4;
          x.w[i] = p[ix]/255.0;
        }
        x = convnetjs.augment(x, 24);

          var label_num = curr_batch_num * batch_size + sample_num;
          return {x:x, label:labels[label_num]};
      }
}


var last_time = 0;
var step = function(sample, sample_num) {

    var x = sample.x;
    var y = sample.label;

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

    var time = new Date().getTime();
    // visualize training status
    if( (time - last_time) > UPDATE_INTERVAL_MS) {

      document.getElementById("fwdTime").innerHTML = 'Forward time per example: ' + stats.fwd_time + 'ms';

      document.getElementById("bpTime").innerHTML = 'Backprop time per example: ' + stats.bwd_time + 'ms';

      document.getElementById("clsLoss").innerHTML = 'Classification loss: ' + f2t(xLossWindow.get_average());

      document.getElementById("exmp").innerHTML = 'L2 Weight decay loss: ' + f2t(wLossWindow.get_average());

      last_time = time;
    }

/*    var t = 'Training accuracy: ' + f2t(trainAccWindow.get_average());
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Validation accuracy: ' + f2t(valAccWindow.get_average());
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));
    var t = 'Examples seen (out of '+ batch_size + "): " + sample_num;
    train_elt.appendChild(document.createTextNode(t));
    train_elt.appendChild(document.createElement('br'));*/

    // visualize activations
/*    if(sample_num % 100 === 0) {
        var vis_elt = document.getElementById("visnet");
        visualize_activations(net, vis_elt);
    }

    // log progress to graph, (full loss)
    if(sample_num % 200 === 0) {
        var xa = xLossWindow.get_average();
        var xw = wLossWindow.get_average();
        if(xa >= 0 && xw >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
            lossGraph.add(step_num, xa + xw);
            lossGraph.drawSelf(document.getElementById("lossgraph"));
        }
    }*/
}
var realPause = false;
var compute = function() {
    paused = !paused;

    var btn = document.getElementById('compute-btn');
    if (paused) {
        $("#compute-btn").text('Train');
        clearInterval(train_on_batch_interval);

            is_training_active = false;
      //  post_gradients_to_server();
    }
    else {
        if(realPause == true) {
            is_training_active = true;
            train_on_batch_interval = setTimeout(train_on_batch, 0);
            return;
        }
        $("#compute-btn").text('Pause');
        init_all();
        start_working();
        realPause=true;
    }
}
// int main
$(window).load(function() {
	client_ID = make_string_ID();
	change_client_name();
	document.getElementById('compute-btn').disabled = false;
});

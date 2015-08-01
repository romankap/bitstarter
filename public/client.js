var UPDATE_INTERVAL_MS = 500;   // GUI text values updates

var net, trainer;
var old_net;

var data_img_elt;
var img_data;

var client_name;

var dataset_name;
var model_id;
var batch_loaded;
var curr_batch_num;

var paused = true;
var started_once = false;

var net_loaded_from_server = false;
var batch_loaded = false;
var training_active = false;
var labels_loaded = false;

var curr_sample_num;

var fw_timings_sum, fw_timings_num;
var bw_timings_sum, bw_timings_num;
var latency_to_server, post_to_server_start, post_to_server_end;
var latency_from_server, get_from_server_start, get_from_server_end;

var last_gui_update = 0;


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

var reset_timing_stats = function() {
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

function isNumeric(num) {
    return !isNaN(num)
}

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
    if(batch_loaded) return;

    data_img_elt = new Image();
    data_img_elt.crossOrigin = 'anonymous';

    data_img_elt.onload = function() {
        var data_canvas = document.createElement('canvas');
        data_canvas.width = data_img_elt.width;
        data_canvas.height = data_img_elt.height;
        var data_ctx = data_canvas.getContext("2d");
        data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
        img_data = data_ctx.getImageData(0, 0, data_img_elt.width, data_img_elt.height);
        batch_loaded = true;
        console.log('Finished loading data batch #' + batch_url);
    }

    data_img_elt.src = batch_url;
}

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var testAccWindow = new cnnutil.Window(50, 1);

var reset_all = function() {    // Just reset any visual cues of doing anything
    net_loaded_from_server = false;
    batch_loaded = false;
    training_active = false;
    curr_sample_num = 0;

    // reinit windows that keep track of val/train accuracies
    xLossWindow.reset();
    wLossWindow.reset();
    trainAccWindow.reset();
    valAccWindow.reset();
    testAccWindow.reset();
}


var init_batch = function() {
    var parameters = {  client_ID: client_ID };

    $.get('/get_net_batch_all', parameters, function(data) {
		    curr_batch_num = data.batch_num;
        curr_epoch_num = data.epoch_num;
        dataset_name = data.dataset_name;

        curr_batch_num = data.batch_num;
        batch_size = data.batch_size;

        load_data_batch(data.batch_url);
		    update_displayed_batch_and_epoch_nums(curr_batch_num, curr_epoch_num, data.total_different_clients);


        net = new convnetjs.Net();
        net.fromJSON(data.net);

        if(model_id != data.model_ID) {
          labels_loaded = false;
          $.getScript(data.dataset_name + '/labels.js', function() {
              labels_loaded = true;
          });
          $("#data_name").text(data.dataset_name.toUpperCase());
        }
        if(trainer == undefined) {
          trainer = new convnetjs.SGDTrainer(net, data.trainer_param);
        } else {
          trainer.net = net;
        }

        model_id  = data.model_ID;

        net_loaded_from_server = true;

        get_from_server_end = new Date().getTime();
        latency_from_server = get_from_server_end - get_from_server_start;
		    reset_timing_stats();



        console.log("<init_all> Received net back. model_ID: " + data.model_ID);
        console.log("<init_all> Working on batch: " + data.batch_num); //DEBUG
      //  console.log("<init_all> Received " + data.net.length + " net in length back"); //DEBUG
    });
}

var post_gradients_to_server = function() {
  training_active = false;
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

var training_setter = function() {
    if (batch_loaded && labels_loaded) {
        training_active = true;
        curr_sample_num = 0;
        setTimeout(train_on_batch, 0);
        console.log('Starting to work!');
    }
    else {
        setTimeout(training_setter, 100);
        console.log('Waiting to start to working...');
    }
}

var launcher = function() {
  if(net_loaded_from_server) {
    setTimeout(training_setter, 0);
  }
  else {
    init_batch();
    setTimeout(training_setter, 0);
  }
}

var post_gradients = function(grads) {
  var parameters = {
                        model_ID: model_id,
                        client_ID: client_ID,
            					  gradients: JSON.stringify(grads)
                      };

  console.log("Sending gradients");

  $.post('/update_model_from_gradients_parts', parameters, function(data) {
	//	console.log(data);
	//	post_to_server_end = new Date().getTime();
	//	latency_to_server = post_to_server_end - post_to_server_start;
	//	console.log("<post_gradients_to_server> Sending latency_to_server: " + latency_to_server);
  });
}

var train_on_batch = function() {
    if (training_active) {
        var sample = sample_training_instance[dataset_name](curr_sample_num);
        step(sample, curr_sample_num); // process this image
        curr_sample_num++

        if(curr_sample_num != 1 && (  curr_sample_num-1) % 80 == 0) {
            grads = trainer.accuredGrads;
            post_gradients(grads);
            trainer.accuredGrads = [];
        }
        if(curr_sample_num == batch_size) {
            net_loaded_from_server = false;
            batch_loaded = false;
          //  post_gradients_to_server();
          //  reset_all();
            if(!paused) {
              setTimeout(launcher, 0);
              return;
            }
        }
        setTimeout(train_on_batch, 0);
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
    if( (time - last_gui_update) > UPDATE_INTERVAL_MS) {

      document.getElementById("fwdTime").innerHTML = 'Forward time per example: ' + stats.fwd_time + 'ms';

      document.getElementById("bpTime").innerHTML = 'Backprop time per example: ' + stats.bwd_time + 'ms';

      document.getElementById("clsLoss").innerHTML = 'Classification loss: ' + f2t(xLossWindow.get_average());

      document.getElementById("decayloss").innerHTML = 'L2 Weight decay loss: ' + f2t(wLossWindow.get_average());

      document.getElementById("accu").innerHTML = 'Training accuracy: ' + f2t(trainAccWindow.get_average());;

      document.getElementById("exmpseen").innerHTML = 'Examples seen (out of '+ batch_size + "): " + sample_num;

      last_gui_update = time;
    }

}

var compute = function() {
    paused = !paused;

    var btn = document.getElementById('compute-btn');
    if (paused) {
        $("#compute-btn").text('Train');
        training_active = false;
      //  post_gradients_to_server();
    }
    else {
    /*    if(started_once) {
            $("#compute-btn").text('Pause');
            start_working();
            return;
        }
        started_once = true;*/
        $("#compute-btn").text('Pause');
        launcher();
    }
}
// int main
$(window).load(function() {
	client_ID = make_string_ID();
	change_client_name();
	document.getElementById('compute-btn').disabled = false;
});

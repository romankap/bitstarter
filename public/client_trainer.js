var net, trainer;
var old_net, curr_net, gradients_net;
var samples_in_batch;
var curr_model_ID, curr_sample_num;
var is_net_loaded_from_server = false, is_training_active = false;
var train_on_batch_interval;
var paused = true;


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

var use_validation_data = true;

function isNumeric(num) {
    return !isNaN(num)
}

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



// loads a training image and trains on it with the network
module.exports = {
  calculate_gradients: function() {
      gradients_calculator.traverse(gradients_net, old_net, "");
  }



var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        return data.batch_num;
    });
}

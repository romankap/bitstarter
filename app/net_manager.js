/**
 * Created by Roman on 05/07/2015.
 */
var convnetjs = require('convnetjs');

function isNumeric(num) {
    return !isNaN(num)
}

function generate_random_number() {
    return Math.floor((Math.random() * 100000) + 1);
}

var gradients_calculator = {
    traverse: function (net_weight, gradient, property_name) {
        for (var i in gradient) {
            if (gradient[i] !== null && typeof(gradient[i]) == "object") {
                //going on step down in the object tree!!
                this.traverse(net_weight[i], gradient[i], property_name + "." + i);
            }
            else if (gradient[i] !== null && typeof(gradient[i]) !== "object" &&
                    net_weight[i] !== null && typeof(net_weight[i]) !== "object" &&
                    net_weight[i] !== NaN && gradient[i] !== NaN &&
                    isNumeric(i) ) {
                net_weight[i] += gradient[i];
            }
        }
    }
};

var add_gradients = function(weights_in_JSON, gradients_in_JSON) {
    gradients_calculator.traverse(weights_in_JSON, gradients_in_JSON, "");
};

var net;          // Most updated netowrk
var dataset;      // Currently used datastet

var model_id;     // Identifier of running model
var last_batch;
var total_batches;
var batch_size;

var trainer_param;

module.exports = {
    get_net: function() {
        return net;
    },

    get_dataset_name: function() {
      return dataset.name;
    },

    get_model_parameters: function() {
        return trainer_param;
    },

    get_batch_size: function () {
        return batch_size;
    },

    get_model_ID: function() {
        return model_id;
    },

    get_train_batch_num: function() {
        return total_batches - 1;
    },

    get_base_model: function() {
        return {
            base_net: net,
            id: model_ID,
            dataset: dataset.name
        };
    },

    get_batch_url: function(batch_num) {
      return dataset.gen_batch_url(batch_num);
    },

    set_net: function(new_net) {
        net = new_net;
    },

    add_gradients: function(update_net) {
        add_gradients(net, update_net);
    },

    update_model_from_gradients: function(model_from_client) {
          var gradients = JSON.parse(model_from_client.net);

          add_gradients(net, JSON.parse(model_from_client.net));
      },

    request_batch_num: function (client) {
        var curr_batch = last_batch;
        console.log("<request_batch_num> sending batch_num " + curr_batch
            + " (out of " + total_batches + ") to node #" + client);
        last_batch++;
        last_batch = last_batch % total_batches;
        return curr_batch;
    },

    reset_batch_num: function () {
        last_batch = 0;
    },

    generate_new_model_ID: function() {
        model_ID = generate_random_number();
        return model_ID;
    },

    get_trainer_param: function() {
        return trainer_param;
    },

    set_dataset: function(dataset_i) {
        dataset = require('./models/' + dataset_i + '.js');

        eval(dataset.init_def);
        model_id = this.generate_new_model_ID();
        total_batches = dataset.total_batches;
        batch_size = dataset.batch_size;
        last_batch = 0;
        trainer_param = {
          learning_rate: trainer.learning_rate,
          momentum: trainer.momentum,
          l2_decay: trainer.l2_decay,
        };
    }
};

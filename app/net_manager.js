/**
 * Created by Roman on 05/07/2015.
 */

function isNumeric(num) {
    return !isNaN(num)
}

module.exports = function (tot_batches) {
    var weights;
    var weights_in_JSON;
    var batch_num = 0;
    var total_batches = tot_batches;

    var increase_batch_num = function () {
        batch_num++;
        batch_num = batch_num % total_batches;

        console.log("<increase_batch_num> new batch_num = " + batch_num + ". Total_batches = " + total_batches);
    };

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

    var functions = {

        store_weights: function(weights_in_JSON_to_store) {
            weights_in_JSON = weights_in_JSON_to_store;
            weights = JSON.stringify(weights_in_JSON);
        },

        update_model_from_gradients: function(model_from_client) {
            trainer.learning_rate = model_from_client.learning_rate;
            trainer.momentum = model_from_client.momentum;
            trainer.l2_decay = model_from_client.l2_decay;

            var gradients = model_from_client.net;
            var gradients_in_JSON = JSON.parse(gradients);

            add_gradients(weights_in_JSON, gradients_in_JSON);
            weights = JSON.stringify(weights_in_JSON);
        },

        get_weights: function() {
            return weights;
        },
        get_model_parameters: function() {
            var params = {learning_rate: trainer.learning_rate, momentum: trainer.momentum, l2_decay: trainer.l2_decay };
            return params;
        },
        get_batch_num: function () {
            return batch_num;
        },
        reset_batch_num: function () {
            batch_num = 0;
        },

        get_and_update_batch_num: function () {
            var curr_batch = batch_num;
            console.log("<get_and_update_batch_num> sending batch_num = " + curr_batch + ". Total_batches = " + total_batches);
            increase_batch_num();
            return curr_batch;
        },
        get_train_batch_num: function() {
            return total_batches - 1;
        }
    };

    return functions;
};
/**
 * Created by Roman on 05/07/2015.
 */

var global_debug_counter=0;

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

        console.log("<increase_batch_num> batch_num = " + batch_num + ". Total_batches = " + total_batches);
    };

    var gradients_calculator = {
        traverse: function (net_weight, gradient, property_name) {

            for (var i in gradient) {

                if (gradient[i] !== null && typeof(gradient[i]) == "object") {
                    //going on step down in the object tree!!
                    this.traverse(net_weight[i], gradient[i], property_name + "." + i);
                }
                else if (gradient[i] !== null && typeof(gradient[i]) !== "object" && isNumeric(i)) {
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

        update_model_from_gradients: function(gradients_from_client) {
            gradients = gradients_from_client;
            gradients_in_JSON = JSON.parse(gradients);
            console.log("<Update model from gradients> Received the gradients" + gradients.substring(0, 2000) + "\n\n");
            console.log("<Update model from gradients> Current weights" + weights.substring(0, 2000));

            add_gradients(weights_in_JSON, gradients_in_JSON);
        },

        get_weights: function() {
            return weights;
        },

        get_batch_num: function () {
            return batch_num;
        },

        get_and_update_batch_num: function () {
            var curr_batch = batch_num;
            increase_batch_num();
            console.log("<get_batch_num> batch_num = " + batch_num + ". Total_batches = " + total_batches);
            return curr_batch;
        },
        get_train_batch_num: function() {
            return total_batches - 1;
        }
    };

    return functions;
};
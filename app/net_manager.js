/**
 * Created by Roman on 05/07/2015.
 */

function isNumeric(num) {
    return !isNaN(num)
}

function generate_random_number() {
    return Math.floor((Math.random() * 100000) + 1);
}

module.exports = function (tot_batches) {
    var total_batches = tot_batches;
    var weights, weights_in_JSON;
    var batch_num = 0;
    var model_ID = 0, init_model;
    var epochs_count = 0;
    var clients_dict = {}, total_different_clients=0, last_contributing_client = "<no client>";

    var increase_batch_num = function () {
        batch_num++;
        if (batch_num === total_batches)
            epochs_count++;

        batch_num = batch_num % total_batches;

        console.log("<increase_batch_num> NEW batch_num = " + batch_num + " (out of " + total_batches + ")");
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

    var check_if_in_clients_dict = function(client_name) {
        if (clients_dict[client_name] !== true) {
            return false;
        }
        return true;
    };
    var insert_to_clients_dict = function(client_name)   {
        if (!check_if_in_clients_dict(client_name)) {
            clients_dict[client_name] = true;
            total_different_clients++;
        }
        last_contributing_client = client_name;
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
            insert_to_clients_dict(model_from_client.client_ID);
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
        get_epochs_count: function () {
            return epochs_count;
        },
        reset_batch_num_and_epochs_count: function () {
            batch_num = 0;
            epochs_count = 0;
        },
        generate_new_model_ID: function() {
            model_ID = generate_random_number();
        },
        get_model_ID: function() {
            return model_ID;
        },

        get_and_update_batch_num: function () {
            var curr_batch = batch_num;
            console.log("<get_and_update_batch_num> sending batch_num = " + curr_batch + " (out of " + total_batches + ")");
            increase_batch_num();
            return curr_batch;
        },
        get_train_batch_num: function() {
            return total_batches - 1;
        },
        store_init_model: function(new_init_model) {
            init_model = new_init_model;
        },
        get_init_model: function() {
            return init_model;
        },
        get_different_clients_num : function() {
            return total_different_clients;
        },
        get_last_contributing_client : function() {
            return last_contributing_client;
        },
        clear_clients_dict : function () {
            clients_dict = {};
            total_different_clients = 0;
            last_contributing_client = "<no client>";
        }

    };

    return functions;
};
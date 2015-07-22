/**
 * Created by Roman on 05/07/2015.
 */
//var os = require('os');


function isNumeric(num) {
    return !isNaN(num)
}

function generate_random_number() {
    return Math.floor((Math.random() * 100000) + 1);
}

var getTime = function() {
    return new Date().getTime();
}

module.exports = function (model_training_batches) {
    var total_training_batches = model_training_batches;
    var weights, weights_in_JSON;
    var batch_num = 0;
    var model_ID = 0, init_model, last_model_ID_sent;
    var epochs_count = 0, last_epoch_sent;
    var clients_dict = {}, total_different_clients=0, last_contributing_client = "<no client>";

    var fw_timings = new Array(), fw_timings_sum=0;
    var bw_timings = new Array(), bw_timings_sum=0;
    var latencies_to_server = new Array(), latencies_to_server_sum=0;
    var latencies_from_server = new Array(), latencies_from_server_sum=0;
    var time_to_train_epochs_array = new Array(), epoch_start_time=0, epoch_end_time=0;
    var validation_accuracies_array = new Array();

    var last_validation_accuracy=0, testing_accuracy=0;

    var increase_batch_num = function () {
        batch_num++;
        batch_num = batch_num % total_training_batches;

        if (batch_num === 0) {
            epochs_count++;
            update_epoch_end_time_and_duration();
        }
        console.log("<increase_batch_num> NEW batch_num = " + batch_num + " (out of " + total_training_batches + ")");
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

    // Stats-related (not needed for export)
    var add_fw_timing = function(fw_timing) {
        fw_timings.push(fw_timing);
        fw_timings_sum += fw_timing;
    }
    var add_bw_timing = function(bw_timing) {
        bw_timings.push(bw_timing);
        bw_timings_sum += bw_timing;
    }
    var add_latencies_to_server = function(latency_to_server){
        if (latency_to_server !== undefined) {
            latencies_to_server.push(latency_to_server);
            latencies_to_server_sum += latency_to_server;
        }
    }

    var update_epoch_start_time = function() {
        epoch_start_time = getTime();
    }
    var update_epoch_end_time_and_duration = function() {
        epoch_end_time = getTime();
        time_to_train_epochs_array.push(epoch_end_time - epoch_start_time);
    }

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
            last_epoch_sent = -1;
            last_model_ID_sent = -1;
        },
        generate_new_model_ID: function() {
            model_ID = generate_random_number();
        },
        get_model_ID: function() {
            return model_ID;
        },

        get_and_update_batch_num: function () {
            var curr_batch = batch_num;
            if (batch_num === 0)
                update_epoch_start_time();

            console.log("<get_and_update_batch_num> sending batch_num = " + curr_batch + " (out of " + total_training_batches + ")");
            increase_batch_num();
            return curr_batch;
        },
        get_train_batch_num: function() {
            return total_training_batches - 1;
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
        },
        is_need_to_send_net_for_testing : function(Admins_model_ID, Admins_epoch_num) {
            if (Admins_model_ID !== model_ID || Admins_epoch_num !== epochs_count) {
                last_model_ID_sent = model_ID;
                last_epoch_sent = epochs_count;
                return true;
            }
            return false;
        },

        // Stats-related
        reset_stats: function() {
            fw_timings.length=0; fw_timings_sum=0;
            bw_timings.length=0; bw_timings_sum=0;
            latencies_to_server.length=0; latencies_to_server_sum=0;
            latencies_from_server.length=0; latencies_from_server_sum=0;
            time_to_train_epochs_array.length=0; epoch_start_time=0; epoch_end_time=0;

            last_validation_accuracy=0; testing_accuracy=0;
        },
        get_fw_timings_average : function() {
            if (fw_timings.length > 0)
                return fw_timings_sum/fw_timings.length;
            return NaN;
        },
        get_bw_timings_average : function() {
            if (bw_timings.length > 0)
                return bw_timings_sum/bw_timings.length;
            return NaN;
        },
        add_latencies_from_server : function(latency_from_server){
            if (latency_from_server !== undefined) {
                latencies_from_server.push(latency_from_server);
                latencies_from_server_sum += latency_from_server;
            }
        },
        get_latencies_from_server_average : function(){
            if (latencies_from_server.length > 0)
                return latencies_from_server_sum/latencies_from_server.length;
            return NaN;
        },
        get_latencies_to_server_average : function(){
            if (latencies_to_server.length > 0)
                return latencies_to_server_sum/latencies_to_server.length;
            return NaN;
        },
        update_stats : function(stats) { //Will be called when client posts the model back to server
            add_fw_timing(stats.fw_timings_average);
            add_bw_timing(stats.bw_timings_average);
            add_latencies_to_server(stats.latency_to_server);
        },
        is_new_validation_accuracy_better : function(new_testing_accuracy, validation_epoch_num){
            validation_accuracies_array[validation_epoch_num] = new_testing_accuracy;

            if (new_testing_accuracy > curr_testing_accuracy) {
                curr_testing_accuracy = new_testing_accuracy;
                return true;
            }
            return false;
        },

        get_stats_in_csv : function() {
            var stats_in_csv="";
            var lines_in_csv = Math.max(fw_timings.length, bw_timings.length,
                            latencies_to_server.length, latencies_from_server.length,
                            time_to_train_epochs_array.length, validation_accuracies_array.length);

            // Headlines
            stats_in_csv += "fw_times" + ",";
            stats_in_csv += "bw_times" + ",";
            stats_in_csv += "latencies_to_server" + ",";
            stats_in_csv += "latencies_from_server" + ",";
            stats_in_csv += "time_to_train_epochs" + ",";
            stats_in_csv += "validation_accuracy";
            stats_in_csv += "\n";

            // Data
            for (var i=0; i<lines_in_csv; i++) {
                if (fw_timings[i] !== undefined)
                    stats_in_csv += fw_timings[i];
                stats_in_csv += ",";

                if (bw_timings[i] !== undefined)
                    stats_in_csv += bw_timings[i];
                stats_in_csv += ",";

                if (latencies_to_server[i]  !== undefined)
                    stats_in_csv += latencies_to_server[i];
                stats_in_csv += ",";

                if (latencies_from_server[i]  !== undefined)
                    stats_in_csv += latencies_from_server[i] ;
                stats_in_csv += ",";

                if (time_to_train_epochs_array[i]  !== undefined)
                    stats_in_csv += time_to_train_epochs_array[i] ;
                stats_in_csv += ",";

                if (validation_accuracies_array[i]  !== undefined)
                    stats_in_csv += validation_accuracies_array[i] ;

                stats_in_csv += "\n";
            }

            return stats_in_csv;
        }
    };

    return functions;
};
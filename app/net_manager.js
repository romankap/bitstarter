/**
 * Created by Roman on 05/07/2015.
 */
var convnetjs = require('convnetjs');

var network_schem;

function isNumeric(num) {
    return !isNaN(num)
}

function generate_random_number() {
    return Math.floor((Math.random() * 100000) + 1);
}

function getTime() {
    return new Date().getTime();
}

function add_gradients(weights_in_JSON, gradients_in_JSON) {
    gradients_calculator.traverse(weights_in_JSON, gradients_in_JSON, "");
}

function update_epoch_start_time() {
	epoch_start_time = getTime();
}

function update_epoch_end_time_and_duration() {
	epoch_end_time = getTime();
	time_to_train_epochs_array.push(epoch_end_time - epoch_start_time);
}
var reset_stats_func = function() {
		fw_timings.length=0; fw_timings_sum=0;
		bw_timings.length=0; bw_timings_sum=0;
		latencies_to_server.length=0; latencies_to_server_sum=0;
		latencies_from_server.length=0; latencies_from_server_sum=0;
		time_to_train_epochs_array.length=0; epoch_start_time=0; epoch_end_time=0;
		validation_accuracies_array.length=0;

    clients_dict = {};
    total_different_clients=0;
    last_contributing_client = "<no client>";

		last_validation_accuracy=0; testing_accuracy=0;
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

var insert_to_clients_dict = function(client_name)   {
	if (!check_if_in_clients_dict(client_name)) {
		clients_dict[client_name] = true;
		total_different_clients++;
	}
	last_contributing_client = client_name;
};

var check_if_in_clients_dict = function(client_name) {
	if (clients_dict[client_name] !== true) {
		return false;
	}
	return true;
};



var net;          // Most updated netowrk
var dataset;      // Currently used datastet

var model_id;     // Identifier of running model
var last_batch;
var total_batches;
var batch_size;

var trainer_param;

var epochs_count = -1, last_epoch_sent;
var clients_dict = {}, total_different_clients=0, last_contributing_client = "<no client>";

var fw_timings = new Array(), fw_timings_sum=0;
var bw_timings = new Array(), bw_timings_sum=0;
var latencies_to_server = new Array(), latencies_to_server_sum=0;
var latencies_from_server = new Array(), latencies_from_server_sum=0;
var time_to_train_epochs_array = new Array(), epoch_start_time=0, epoch_end_time=0;
var validation_accuracies_array = new Array();

var last_validation_accuracy=0, testing_accuracy=0;

module.exports = {

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

	get_epochs_count: function () {
        return epochs_count;
    },

	// Stats-related
	reset_stats: reset_stats_func,

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
	is_new_validation_accuracy_worse : function(new_testing_accuracy, validation_epoch_num){
		validation_accuracies_array[validation_epoch_num] = new_testing_accuracy;

		if (last_validation_accuracy <= new_testing_accuracy) {
			last_validation_accuracy = new_testing_accuracy;
			return false;
		}
		return true;
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
	},

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

    get_base_model_data: function() {
        return {
            base_net: net.toJSON(),
            model_ID: model_id,
            dataset: dataset.name,

      			total_training_batches: 		dataset.train_batches,
      			samples_in_training_batch: 		dataset.train_size,
      			samples_in_testing_batch: 		dataset.test_size,
      			samples_in_validation_batch: 	dataset.validation_size,
      			minimum_epochs_to_train: 		dataset.minimum_epochs_to_train,
        };
    },

    get_batch_url: function(batch_num) {
      return dataset.gen_batch_url(batch_num, batch_size);
    },

    get_current_net_schem: function() {
        return network_schem;
    },

    set_batch_size: function(val) {
        batch_size = val;
    },

    set_net: function(new_net) {
        network_schem = new_net;

        model_id = generate_random_number();
        last_batch = 0;
        epochs_count = 0;


        eval(network_schem);
        trainer_param = {
              learning_rate: trainer.learning_rate,
              momentum: trainer.momentum,
              l2_decay: trainer.l2_decay,
              l1_decay: trainer.l1_decay,
              method: trainer.method,
              batch_size: trainer.batch_size
        };

        is_model_in_testing_mode = false;
    },

    add_gradients: function(update_net) {
        add_gradients(net, update_net);
    },

    gen_admin_batch_url: function() {
      return dataset.admin_url(batch_size);
    },

    update_model_from_gradients: function(model_from_client) {
          var gradients = JSON.parse(model_from_client.net);

          add_gradients(net, JSON.parse(model_from_client.net));
		  insert_to_clients_dict(model_from_client.client_ID);
      },

	get_batch_num:	function() {
		return last_batch;
	},

    request_batch_num: function (client) {
        var curr_batch = last_batch;
        console.log("<request_batch_num> sending batch_num " + curr_batch
            + " (out of " + total_batches + ") to node #" + client);
        last_batch++;
        last_batch = last_batch % total_batches;
		if(curr_batch == 0) {
			epochs_count++;
			if(epochs_count != 0) {
				update_epoch_end_time_and_duration();
			}
			update_epoch_start_time();
		}
        return curr_batch;
    },


    reset_batch_num: function () {
        last_batch = 0;
    },

	reset_model: function() {
		model_ID = generate_random_number();
		last_batch = 0;
		epochs_count = 0;

    reset_stats_func();
		eval(network_schem);
		trainer_param = {
          learning_rate: trainer.learning_rate,
          momentum: trainer.momentum,
          l2_decay: trainer.l2_decay,
          l1_decay: trainer.l1_decay,
          method: trainer.method,
          batch_size: trainer.batch_size
    };

		is_model_in_testing_mode = false;
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

        network_schem = dataset.init_def;

        eval(network_schem);
        model_id = this.generate_new_model_ID();
        total_batches = dataset.train_batches;
        batch_size = dataset.train_size;
        last_batch = 0;
		    epochs_count = 0;

        trainer_param = {
              learning_rate: trainer.learning_rate,
              momentum: trainer.momentum,
              l2_decay: trainer.l2_decay,
              l1_decay: trainer.l1_decay,
              method: trainer.method,
              batch_size: trainer.batch_size
        };
    }
};

/**
 * Created by Roman on 04/07/2015.
 */
var convnetjs = require('convnetjs');
var net_manager = require('../net_manager');

/// ---- DEBUG ---- ///

var total_cifar10_training_samples = 100;
var samples_in_training_batch=50;
var samples_in_test_batch=100;
var samples_in_validation_batch=100;

var total_training_batches = total_cifar10_training_samples / samples_in_training_batch;
var minimum_epochs_to_train=2;

/*var total_cifar10_training_samples = 45000;
var samples_in_training_batch=1000;
var samples_in_test_batch=1000;
var samples_in_validation_batch=1000;

var total_training_batches = total_cifar10_training_samples / samples_in_training_batch;
var minimum_epochs_to_train=50;*/

exports.total_training_batches = total_training_batches;
exports.samples_in_training_batch = samples_in_training_batch;
exports.samples_in_testing_batch = samples_in_test_batch;
exports.samples_in_validation_batch = samples_in_validation_batch;
exports.minimum_epochs_to_train = minimum_epochs_to_train;

/////////////////////////////////////////////

var cifar10_manager = net_manager(total_training_batches);

var cifar10_init_model = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});\n\
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:2, stride:2});\n\
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:2, stride:2});\n\
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:2, stride:2});\n\
layer_defs.push({type:'softmax', num_classes:10});\n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});\n\
";

cifar10_manager.store_init_model(cifar10_init_model);


//exports.convnet = convnet.convnetjs;
var reset_model =  function () {
    eval(cifar10_manager.get_init_model());
    cifar10_manager.store_weights(net.toJSON());
    cifar10_manager.reset_batch_num_and_epochs_count();
    cifar10_manager.clear_clients_dict();
    cifar10_manager.generate_new_model_ID();

    cifar10_manager.reset_stats();
};

var init_new_model = function (new_init_model) {
    cifar10_manager.store_init_model(new_init_model);
    reset_model();
};
reset_model();

exports.reset_model = reset_model;
exports.init_new_model = init_new_model;
exports.net_manager = cifar10_manager;

//console.log("<<< The convnet in JSON >>>");
//console.log(net.toJSON());
//console.log(JSON.stringify(net.toJSON()).length);

exports.convnetjs = convnetjs;
exports.net = net;
/**
 * Created by Roman on 04/07/2015.
 */

var convnetjs = require('convnetjs');
var net_manager = require('../net_manager');

var total_batches = 50;
var cifar10_manager = net_manager(total_batches);


var init_model = "layer_defs = [];\n\
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

exports.init_model = init_model;

//exports.convnet = convnet.convnetjs;

eval(init_model);
cifar10_manager.store_weights(JSON.stringify(net.toJSON()));

exports.net_manager = cifar10_manager;

//console.log("<<< The convnet in JSON >>>");
//console.log(net.toJSON());
//console.log(JSON.stringify(net.toJSON()).length);

exports.convnetjs = convnetjs;
exports.net = net;
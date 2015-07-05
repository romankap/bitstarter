/**
 * Created by Roman on 04/07/2015.
 */

var net_manager = require('../net_manager');

var init_model = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});\n\
layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:2, stride:2});\n\
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:3, stride:3});\n\
layer_defs.push({type:'softmax', num_classes:10});\n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});\n\
";

exports.init_model = init_model;
exports.net_manager = net_manager;
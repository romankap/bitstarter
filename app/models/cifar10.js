/**
 * Created by Roman on 04/07/2015.
 */

//var cnnutil = require('../../public/build/util');
//var convnet = require('../../public/build/convnet');

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

//eval(init_model);

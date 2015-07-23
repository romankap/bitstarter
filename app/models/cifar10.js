/**
 * Created by Roman on 04/07/2015.
 */

module.exports = {
  name: "cifar10",
  
  train_size: 500,
  train_batches: 45000 / 500,
  validation_size: 5000,
  test_size: 10000,
  
  minimum_epochs_to_train: 50,

  init_def: "layer_defs = [];\n\
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
  trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:3, l2_decay:0.0001});\n",

  gen_batch_url: function(batch) {
    return "http://tx.technion.ac.il/~sromanka/cifar10/500/cifar10_batch_" + batch + ".png";;
  }
};

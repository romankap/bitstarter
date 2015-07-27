module.exports = {
	name: "mnist",

	train_size: 500,
	train_batch_size: 50000,
	validation_size: 10000,
	test_size: 10000,

	minimum_epochs_to_train: 10,

	init_def: "layer_defs = [];\n\
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
	trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:2, l2_decay:0.001});",

	gen_batch_url: function(batch, batch_size) {
		return "http://tx.technion.ac.il/~sromanka/mnist/" + batch_size + "/mnist_batch_" + batch + ".png";
	},

	admin_url: function(batch, batch_size) {
		return "http://tx.technion.ac.il/~sromanka/mnist/500/mnist_admin.png"
	},
};

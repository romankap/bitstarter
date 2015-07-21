var net, trainer;
var client_ID, curr_batch_num=-1, curr_epoch_num;
var num_batches = 51; // 50 training batches, 1 test
var test_batch = 50;
var data_img_elt; //TODO: make this a single element instead of an array
var img_data// = new Array(num_batches);

var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var testAccWindow = new cnnutil.Window(50, 1);

var is_batch_loaded;// = new Array(num_batches);
//var loaded_train_batch = [];
var init_model;
// ------------------------
// BEGIN CIFAR10 SPECIFIC STUFF
// ------------------------
var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

var use_validation_data = true;

// Returns a random number in the range: [0, max_num-1]
var get_random_number = function (max_num) {
    return Math.floor((Math.random() * max_num));
}

var make_string_ID = function ()
{
    var text = "";
    var consonants = "AEIO";
    var vowels = "BDFGJKLMNPRSTVYZ";

    for( var i=0; i < 5; i++ ) {
        if (i % 2 == 0)
            text += vowels.charAt(Math.floor(Math.random() * vowels.length));
        else
            text += consonants.charAt(Math.floor(Math.random() * consonants.length));
    }

    return text;
}

var change_client_name = function() {
    $('#client-name').html('Hello trainer-client <span class=\"color-blue\">' + client_ID + "</span>");
    $('#client-name-explanation').html("(" + client_ID + " is your identifying name in the crowdcomputing network)");
}

var update_displayed_batch_and_epoch_nums = function(new_batch_num, new_epoch_num) {
    if (new_batch_num === -1)
        $('#batch-num').html("Waiting for batch and epoch to update");
    else
        $('#batch-num').html("Training on batch #" + new_batch_num + " , epoch #" + new_epoch_num);
}

var load_data_batch = function(batch_to_load) {
    // Load the dataset with JS in background
    is_batch_loaded = false;
    data_img_elt = new Image();
    data_img_elt.crossOrigin = 'anonymous';

    data_img_elt.onload = function() {
        var data_canvas = document.createElement('canvas');
        data_canvas.width = data_img_elt.width;
        data_canvas.height = data_img_elt.height;
        var data_ctx = data_canvas.getContext("2d");
        data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
        img_data = data_ctx.getImageData(0, 0, data_img_elt.width, data_img_elt.height);
        is_batch_loaded = true;
        console.log('finished loading data batch ' + batch_to_load);
    }
    data_img_elt.src = "https://s3.eu-central-1.amazonaws.com/bitstarter-dl/cifar10/cifar10_batch_" + batch_to_load + ".png";
}

// ------------------------
// END MNIST SPECIFIC STUFF
// ------------------------

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {

    var s = scale || 2; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    // get max and min activation to scale the maps automatically
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    // create the canvas elements, draw and add to DOM
    for(var d=0;d<A.depth;d++) {

        var canv = document.createElement('canvas');
        canv.className = 'actmap';
        var W = A.sx * s;
        var H = A.sy * s;
        canv.width = W;
        canv.height = H;
        var ctx = canv.getContext('2d');
        var g = ctx.createImageData(W, H);

        for(var x=0;x<A.sx;x++) {
            for(var y=0;y<A.sy;y++) {
                if(draw_grads) {
                    var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
                } else {
                    var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);
                }
                for(var dx=0;dx<s;dx++) {
                    for(var dy=0;dy<s;dy++) {
                        var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                        for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
                        g.data[pp+3] = 255; // alpha channel
                    }
                }
            }
        }
        ctx.putImageData(g, 0, 0);
        elt.appendChild(canv);
    }
}

var draw_activations_COLOR = function(elt, A, scale, grads) {

    var s = scale || 2; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    // get max and min activation to scale the maps automatically
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);
    for(var d=0;d<3;d++) {
        for(var x=0;x<A.sx;x++) {
            for(var y=0;y<A.sy;y++) {
                if(draw_grads) {
                    var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
                } else {
                    var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);
                }
                for(var dx=0;dx<s;dx++) {
                    for(var dy=0;dy<s;dy++) {
                        var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                        g.data[pp + d] = dval;
                        if(d===0) g.data[pp+3] = 255; // alpha channel
                    }
                }
            }
        }
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
}

var visualize_activations = function(net, elt) {

    // clear the element
    elt.innerHTML = "";

    // show activations in each layer
    var N = net.layers.length;
    for(var i=0;i<N;i++) {
        var L = net.layers[i];

        var layer_div = document.createElement('div');

        // visualize activations
        var activations_div = document.createElement('div');
        activations_div.appendChild(document.createTextNode('Activations:'));
        activations_div.appendChild(document.createElement('br'));
        activations_div.className = 'layer_act';
        var scale = 2;
        if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax

        // HACK to draw in color in input layer
        if(i===0) {
            draw_activations_COLOR(activations_div, L.out_act, scale);
            draw_activations_COLOR(activations_div, L.out_act, scale, true);

            /*
             // visualize positive and negative components of the gradient separately
             var dd = L.out_act.clone();
             var ni = L.out_act.w.length;
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq > 0 ? dwq : 0.0; }
             draw_activations_COLOR(activations_div, dd, scale);
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq < 0 ? -dwq : 0.0; }
             draw_activations_COLOR(activations_div, dd, scale);
             */

            /*
             // visualize what the network would like the image to look like more
             var dd = L.out_act.clone();
             var ni = L.out_act.w.length;
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] -= 20*dwq; }
             draw_activations_COLOR(activations_div, dd, scale);
             */

            /*
             // visualize gradient magnitude
             var dd = L.out_act.clone();
             var ni = L.out_act.w.length;
             for(var q=0;q<ni;q++) { var dwq = L.out_act.dw[q]; dd.w[q] = dwq*dwq; }
             draw_activations_COLOR(activations_div, dd, scale);
             */

        } else {
            draw_activations(activations_div, L.out_act, scale);
        }

        // visualize data gradients
        if(L.layer_type !== 'softmax' && L.layer_type !== 'input' ) {
            var grad_div = document.createElement('div');
            grad_div.appendChild(document.createTextNode('Activation Gradients:'));
            grad_div.appendChild(document.createElement('br'));
            grad_div.className = 'layer_grad';
            var scale = 2;
            if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax
            draw_activations(grad_div, L.out_act, scale, true);
            activations_div.appendChild(grad_div);
        }

        // visualize filters if they are of reasonable size
        if(L.layer_type === 'conv') {
            var filters_div = document.createElement('div');
            if(L.filters[0].sx>3) {
                // actual weights
                filters_div.appendChild(document.createTextNode('Weights:'));
                filters_div.appendChild(document.createElement('br'));
                for(var j=0;j<L.filters.length;j++) {
                    // HACK to draw in color for first layer conv filters
                    if(i===1) {
                        draw_activations_COLOR(filters_div, L.filters[j], 2);
                    } else {
                        filters_div.appendChild(document.createTextNode('('));
                        draw_activations(filters_div, L.filters[j], 2);
                        filters_div.appendChild(document.createTextNode(')'));
                    }
                }
                // gradients
                filters_div.appendChild(document.createElement('br'));
                filters_div.appendChild(document.createTextNode('Weight Gradients:'));
                filters_div.appendChild(document.createElement('br'));
                for(var j=0;j<L.filters.length;j++) {
                    if(i===1) { draw_activations_COLOR(filters_div, L.filters[j], 2, true); }
                    else {
                        filters_div.appendChild(document.createTextNode('('));
                        draw_activations(filters_div, L.filters[j], 2, true);
                        filters_div.appendChild(document.createTextNode(')'));
                    }
                }
            } else {
                filters_div.appendChild(document.createTextNode('Weights hidden, too small'));
            }
            activations_div.appendChild(filters_div);
        }
        layer_div.appendChild(activations_div);

        // print some stats on left of the layer
        layer_div.className = 'layer ' + 'lt' + L.layer_type;
        var title_div = document.createElement('div');
        title_div.className = 'ltitle'
        var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
        title_div.appendChild(document.createTextNode(t));
        layer_div.appendChild(title_div);

        if(L.layer_type==='conv') {
            var t = 'filter size ' + L.filters[0].sx + 'x' + L.filters[0].sy + 'x' + L.filters[0].depth + ', stride ' + L.stride;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }
        if(L.layer_type==='pool') {
            var t = 'pooling size ' + L.sx + 'x' + L.sy + ', stride ' + L.stride;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }

        // find min, max activations and display them
        var mma = maxmin(L.out_act.w);
        var t = 'max activation: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
        layer_div.appendChild(document.createTextNode(t));
        layer_div.appendChild(document.createElement('br'));

        var mma = maxmin(L.out_act.dw);
        var t = 'max gradient: ' + f2t(mma.maxv) + ', min: ' + f2t(mma.minv);
        layer_div.appendChild(document.createTextNode(t));
        layer_div.appendChild(document.createElement('br'));

        // number of parameters
        if(L.layer_type==='conv' || L.layer_type==='local') {
            var tot_params = L.sx*L.sy*L.in_depth*L.filters.length + L.filters.length;
            var t = 'parameters: ' + L.filters.length + 'x' + L.sx + 'x' + L.sy + 'x' + L.in_depth + '+' + L.filters.length + ' = ' + tot_params;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }
        if(L.layer_type==='fc') {
            var tot_params = L.num_inputs*L.filters.length + L.filters.length;
            var t = 'parameters: ' + L.filters.length + 'x' + L.num_inputs + '+' + L.filters.length + ' = ' + tot_params;
            layer_div.appendChild(document.createTextNode(t));
            layer_div.appendChild(document.createElement('br'));
        }

        // css madness needed here...
        var clear = document.createElement('div');
        clear.className = 'clear';
        layer_div.appendChild(clear);

        elt.appendChild(layer_div);
    }
}




// user settings 
var change_lr = function() {
    trainer.learning_rate = parseFloat(document.getElementById("lr_input").value);
    update_net_param_display();
}
var change_momentum = function() {
    trainer.momentum = parseFloat(document.getElementById("momentum_input").value);
    update_net_param_display();
}
var change_batch_size = function() {
    trainer.batch_size = parseFloat(document.getElementById("batch_size_input").value);
    update_net_param_display();
}
var change_decay = function() {
    trainer.l2_decay = parseFloat(document.getElementById("decay_input").value);
    update_net_param_display();
}
var update_net_param_display = function() {
    document.getElementById('lr_input').value = trainer.learning_rate;
    document.getElementById('momentum_input').value = trainer.momentum;
    document.getElementById('batch_size_input').value = trainer.batch_size;
    document.getElementById('decay_input').value = trainer.l2_decay;
}


var dump_json = function() {
    document.getElementById("dumpjson").value = JSON.stringify(this.net.toJSON());
}
var clear_graph = function() {
    lossGraph = new cnnvis.Graph(); // reinit graph too
}
var reset_all = function() {
    // reinit trainer
    trainer = new convnetjs.SGDTrainer(net, {learning_rate:trainer.learning_rate, momentum:trainer.momentum, batch_size:trainer.batch_size, l2_decay:trainer.l2_decay});
    update_net_param_display();

    // reinit windows that keep track of val/train accuracies
    xLossWindow.reset();
    wLossWindow.reset();
    trainAccWindow.reset();
    valAccWindow.reset();
    testAccWindow.reset();
    lossGraph = new cnnvis.Graph(); // reinit graph too
    step_num = 0;
}
var load_from_json = function(jsonString) {
    var json = JSON.parse(jsonString);
    net = new convnetjs.Net();
    net.fromJSON(json);
    reset_all();
}

var load_pretrained = function() {
    $.getJSON("cifar10/cifar10_snapshot.json", function(json){
        net = new convnetjs.Net();
        net.fromJSON(json);
        trainer.learning_rate = 0.0001;
        trainer.momentum = 0.9;
        trainer.batch_size = 2;
        trainer.l2_decay = 0.00001;
        reset_all();
    });
}

var get_batch_num_from_server = function() {
    var parameters = {model_name: "CIFAR10" };
    $.get('/get_batch_num_from_server', parameters, function(data) {
        console.log("<get_batch_num_from_server> Starting to work on batch_num: " + data.batch_num);
        return data.batch_num;
    });
}

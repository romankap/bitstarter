/**
 * Created by Roman on 05/07/2015.
 */

module.exports = function (tot_batches) {
    var weights;
    var batch_num = 0;
    var total_batches = tot_batches;

    var increase_batch_num = function () {
        batch_num++;
        batch_num = batch_num % total_batches;

        console.log("<increase_batch_num> batch_num = " + batch_num + ". Total_batches = " + total_batches);
    };

    var functions = {
        store_weights: function(weights_in_JSON) {
            weights = weights_in_JSON;
        },

        get_weights: function() {
            return weights;
        },

        get_batch_num: function () {
            var curr_batch = batch_num;
            increase_batch_num();
            console.log("<get_batch_num> batch_num = " + batch_num + ". Total_batches = " + total_batches);
            return curr_batch;
        },
        get_train_batch_num: function() {
            return total_batches - 1;
        }
    };

    return functions;
};
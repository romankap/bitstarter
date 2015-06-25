var mongoose = require('mongoose');

var Schema = mongoose.Schema;

var UserSchema = new Schema({
    name: String,
    username: { type: String, required: true, index: {unique: true}},
    password: { type: String, required: true, select: false}
});

UserSchema.pre('save', function(next) {
    
    var user = this;
    
    if (!user.isModified('password'))
        return next();
    
    
})

UserSchema.methods.comparePassword = function(password) {
    var user = this;
    
    return true;
}

module.exports = mongoose.model('User', UserSchema);
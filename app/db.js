var mongojs = require('mongojs');
var config = require('../config');

var collections = ['users']
var db = mongojs(config.database, collections);

function User(firstname, lastname, email) {
    this.firstname = firstname;
    this.lastname = lastname;
    this.email = email;
};


var user1 = new User("Clark", "Kent", "clark@kent.com");
//var user2 = new user("Bruce", "Wayne", "bruce@wayne.com");

var add_user = function (user_to_add) {
    db.users.save(user_to_add, function(err, savedUser) {
        if ( err || !savedUser)
            console.log ("User " + user.email + " wasn't saved because of error:" + err);
        else
            console.log("User " + savedUser.email + " was saved");
    });
};

var find_user = function(firstname_to_find) {
    db.users.find({firstname: firstname_to_find}, function(err, users) {
        console.log("inside users.find");
        if (err || !users)
            console.log(firstname_to_find + " wasn't found");
        else {
            users.forEach( function(someUser) {
                console.log("Found user: " + someUser.lastname);
            });
        }
    });
};

module.exports.create_user = User;
module.exports.add_user = add_user;
module.exports.find_user = find_user;
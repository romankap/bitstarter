var express = require('express');
var bodyParser = require('body-parser');
var morgan = require('morgan');
var http = require('http');
var fs = require('fs');
//var mongoose = require('mongoose');
var config = require('./config');
var db = require('./app/db');

var app = express();

var user = new db.create_user('Roman', 'Kaplan', "some@one.com");
console.log("created user:" + user.firstname);
db.add_user(user);
db.find_user('Roman');

// Serve HTTP requests
app.set('port', (process.env.PORT || 8080))
app.use(express.static(__dirname + '/public'))

app.get('/', function(request, response) {
    //fs.readFileSync("index.html");
    var index_buffer = new Buffer(fs.readFileSync("index.html"))
    response.send(index_buffer.toString())
});

app.listen(app.get('port'), function() {
  console.log("Node app is running at localhost:" + app.get('port'))
});
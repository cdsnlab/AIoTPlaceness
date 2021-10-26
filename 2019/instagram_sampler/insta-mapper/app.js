const express = require('express');
const app = express();
const mongoose = require('mongoose');
const port = 3333;

app.set('view engine', 'jade')
app.set('views', './views')

mongoose.connect('mongodb://143.248.55.32/placeness');
var nullSchema = new mongoose.Schema({});
const Posts = mongoose.model('Posts', nullSchema);
const Images = mongoose.model('Images', nullSchema);

app.use(express.static('.'));

app.get('/', function (req, res) {
	res.sendfile('sample.html');
});

app.get('/redirect', function (req, res) {
	res.render('redirect');
});

app.get('/location/:loc_id', function (req, res) {
	Posts.find({'loc_id': req.params.loc_id}, function(err, data){
		if (err) res.send(err);
		else{
			if (data.length == 0){
				res.redirect('/redirect');
			}
			else{
				var rdata = [];
				data.forEach(function(item){
					var json_item = item.toJSON();
					var ritem = {caption: json_item.caption, images: []};
					for (var i in json_item.children){
						ritem.images.push(json_item.file_path + json_item.children[i]);
					}
					rdata.push(ritem);
				});
				//console.log(rdata);
				res.render('location', {'_title': req.params.loc_id, 'posts':rdata});
			}
		}
	});
});

app.listen(port, function () {
	console.log('Example app listening on port', port);
});


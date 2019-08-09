const express = require('express')
const fs = require('fs');
const app = express();
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const config = require('./config.js');
const port = config.port;
const TagDB = config.tagdb;
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended : true}));

mongoose.connect('mongodb://localhost/placeness');
var nullSchema = new mongoose.Schema({});
var Posts = mongoose.model('Posts', nullSchema);
var Images = mongoose.model('Images', nullSchema);
var tagSchema = new mongoose.Schema({
	'image_shortcode': {type: String, index: true, unique: true},
	date: {type: Date, default: Date.now },
	tags: Array(),
	image_url: String
});
var Tags = mongoose.model(TagDB, tagSchema);

var count = 170000;
function getRandomInt() {
	return Math.floor(Math.random() * (count - 0));
}
function ontologyInterpreter(text_ontology){
	var root = {text: { name: "Activity"}, children: [] , HTMLclass: 'black'};
	var last_parent = {};
	var lines = text_ontology.split('\n')
	var maxlvl = 0;
	for (var i in lines){
		var item = lines[i];
		var lvl = 0;
		for (var i = 0; i < item.length; i++){
			if (item[i] == '~') lvl += 1;
			else break;
		}

		if (!last_parent.hasOwnProperty(lvl)){
			last_parent[lvl] = 0;
			if (maxlvl < lvl){
				maxlvl = lvl;
			}
		}

		ont_name = item.substring(lvl);
		var alias = root;
		var orderName = '';
		for(var t = 0; t < lvl; t++){
			alias = alias.children[last_parent[t]-1];
			orderName += last_parent[t] + ".";
		}
		last_parent[lvl] += 1;
		for(var t = lvl+1; t <= maxlvl; t++){
			last_parent[t] = 0;
		}

		orderName += last_parent[lvl] + '|';

		alias.children.push({
			text: {name: orderName + ont_name},
			children: []
		});


	}
	return root;
}

var ridx = getRandomInt();
var history = [ridx];
var hidx = 0;

app.set('view engine', 'jade')
app.set('views', './views')

app.use('/public', express.static('public'))
app.use('/super-simple', express.static('super-simple'))
app.use('/images', express.static('/E/images'))
//app.use('/20180301~20180930', express.static('/E/20180301~20180930'))
//app.use('/20180931~', express.static('/E/20180931~'))
//app.get('/', (req, res) => res.sendfile('super-simple/index.html'))
app.get('/', function(req, res){
	Images.countDocuments({}, function(err, icount){
		if (err){
			res.send(err);
		}
		else{
			count = icount;
			//res.redirect('/template');
			res.render('main', {"_title":"Main"});
		}
	});
});


//var contents = fs.readFileSync('/E/tot_jpg_list.txt', 'utf8');
//contents = contents.trim().split('\n');
var text_ontology = fs.readFileSync('./place-ontology.txt', 'utf8');
ontology = text_ontology.trim().split('\n');


/*
app.get('/reset', function (req, res) {
	ridx = getRandomInt(0, contents.length);
	//history.push(ridx);
	//hidx += 1;

	history[hidx] = ridx;
	res.redirect('/template');
});
*/
app.get('/prev', function (req, res) {
	if (hidx > 0){
		hidx -= 1;
	}
	ridx = history[hidx];
	res.redirect('/template');
});
app.get('/next', function (req, res) {
	hidx += 1;
	if ( history.length > hidx ){
		ridx = history[hidx];
	}
	else {
		ridx = getRandomInt();
		history.push(ridx);
	}
	res.redirect('/template');
});
/*
app.get('/random', function (req, res) {
	res.sendfile('/E/'+contents[ridx]);
});
*/
app.get('/template', function(req, res) {
	//var text_ontology = fs.readFileSync('./place-ontology.txt', 'utf8');
	var root = ontologyInterpreter(text_ontology);
	Images.findOne({'image_local_id': ridx}, function(err, image_data){
		if (err) {
			res.send("Fail");
		}
		else {
			image_data = image_data.toJSON();
			Tags.findOne({'image_shortcode': image_data.image_shortcode}, function(err, tag_data){
				var mtags = [];
				if (err) return;
				if (tag_data){
					mtags = tag_data.toJSON().tags;
				}
				
				
				var tag_tokens = mtags;
				for (var tagidx in tag_tokens){
					var tag = tag_tokens[tagidx];
					var alias = root;

					var label_tokens = tag.split('|');
					for (var labelidx in label_tokens){
						var mlabel = label_tokens[labelidx];
						var unfound = true;
						for (var catidx in alias.children){
							var cat = alias.children[catidx];
							var cat_token = cat.text.name.split('|')
							var cat_order = cat_token[0];
							var cat_name = cat_token[1];
							console.log(mlabel, cat_name);
							if (cat_name == mlabel){
								unfound = false;
								alias = alias.children[catidx];
								break;
							}
						}
						if (unfound){
							console.log("error");
						}
					}
					console.log(alias);
					alias.HTMLclass = 'switch-on';
				}

				var image_url = image_data.image_path;

				res.render('temp', {
					_title: "Template", 
					tree: root, 
					image_shortcode: image_data.image_shortcode,
					tags: mtags,
					image_url: image_url
				});
			});
		}
	});
});
app.post('/save_ontology', function(req, res){
	text_ontology = req.body.data;
	fs.writeFileSync('./place-ontology.txt', text_ontology);
	ontology = text_ontology.trim().split('\n');
	res.send("Success");
});
app.post('/save_placeness', function(req, res){
	var query_code = req.body.image_shortcode;
	var insert_tags = req.body.tags;
	var image_url = req.body.image_url;
	console.log("code:", query_code);
	Tags.findOne({'image_shortcode': query_code}, function(err, data){
		if (err) {
		}
		else {
			if (data == null){
				var cat = new Tags({'image_shortcode': query_code, 'tags': insert_tags, 'image_url': image_url});
				cat.save(function(err, data){
					if(err){
						console.log("Error in creating");
						res.send("Error in creating");
					}
					else{
						console.log(data);
						console.log("Newly created");
						res.send("Newly created");
					}
				});
			}
			else{
				Tags.updateOne({'image_shortcode': query_code}, {'tags': insert_tags},
					function(err, data){
						if(err){
							console.log("Error in updating");
							res.send("Error in updating");
						}
						else{
							console.log("Updated:", data);
							res.send("Updated");
						}
					}
				);
			}
		}
	});
});

app.get('/samplePost', function(req, res){
	Posts.findOne(function(err, data){
		if (err){
			res.send(err);
		}
		else {
			res.send(data.toJSON());
		}
	});
});
app.get('/random', function(req, res){
	Images.findOne({'image_local_id': ridx}, function(err, data){
		if (err){
			res.send(err);
		}
		else {
			//res.send(data);
			json_data = data.toJSON();
			res.sendfile("/E/" + json_data['parent_dir']
						  + '/%25' + json_data['loc_id']
						  + '/' + json_data['image_path']);
		}
	});
});
app.get('/ontology', (req, res) => {
	text_ontology = fs.readFileSync('./place-ontology.txt', 'utf8');
	var root = ontologyInterpreter(text_ontology);
	res.render('ontology', {'text': text_ontology, 'tree': root})
}); 
app.get('/search', (req, res) => {
	var root = ontologyInterpreter(text_ontology);
	res.render('search', {
		_title: "Search", 
		tree: root, 
	});
}); 
app.post('/search_placeness', function(req, res){
	var query_tags = req.body.tags;
	console.log("query_tags:", query_tags);
	Tags.find({'tags': {$all: query_tags}}, function(err, data){
		if (err) {
			res.send(err);
		}
		else {
			res.send(data);
		}
	});
});
app.post('/delete_placeness', function(req, res){
	var image_shortcode = req.body.image_shortcode;
	var delete_tag = req.body.delete_tag;
	Tags.findOne({'image_shortcode': image_shortcode}, function(err, data){
		if (err) {
			res.send(err);
		}
		else {
			console.log(data);
			var mlist = data.tags;
			mlist.splice( mlist.indexOf(delete_tag), 1 );

			Tags.updateOne({'image_shortcode': image_shortcode}, {'tags': mlist},
				function(err, data){
					if(err){
						console.log("Error in updating");
						res.send("Error in updating");
					}
					else{
						console.log("Updated:", data);
						res.send("Updated");
					}
				}
			);
		}
	});
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))

$(function(){
	var simple_chart_config = {
		chart: {
			container: '#tree-simple',
			levelSeparation: 45,
			rootOrientation: "WEST",
			nodeAlign: "BOTTOM",
			connectors: {
				type: "step",
				style: {
					"stroke-width": 2
				}
			},
		},
		nodeStructure: nStruct
	};


	var chart = new Treant(simple_chart_config);
	$('.node').click(function(){
		console.log($(this).text());
		if('Activity' == $(this).text()) return;
		$('.switch-on').each(function(item){
			$(this).removeClass('switch-on');
		});
		$(this).addClass('switch-on');

		var post_data = {
			tags: []
		}

		$('.switch-on').each(function(index, item){
			console.log(post_data);
			var name = $(this).text();
			//console.log(name);

			var tokens = name.split('|');
			if (tokens.length < 2){
				return;
			}
			var label = tokens[0];
			var alias = nStruct;
			//var tag = tokens[1];
			var tag = '';
			label.split('.').forEach(function(element){
				if (tag.length > 0){
					tag += '|';
				}
				//console.log(element);
				alias = alias.children[element-1];
				tag += alias.text.name.split('|')[1];
				console.log(alias.text.name);
			});
			post_data.tags.push(tag);
		});

		$.post('/search_placeness',
			post_data,
			function(result){
				console.log(result);
				$('#target-result').html('');
				result.forEach(function(item){
					var mimg = $('<img>');
					//$('#target-result').append('<div style="display:inline"><img shortcode="'+item.image_shortcode+'"class="small-img" src="http://143.248.55.32/' + item.image_url + '"></img></div>');
					mimg.attr('src', 'http://smhanlab.com/insta-img/' + item.image_url);
					mimg.addClass('small-img');
					var mbtn = $('<button>');
					/*mbtn.attr('onclick', 'func_close('+item.image_shortcode+')');*/
					mbtn.addClass('m-close');
					mbtn.append($('<span>').html('&times'));
					mbtn.attr('shortcode', item.image_shortcode);
					var mdiv = $('<div style="display:inline">');
					mdiv.append(mimg);
					mdiv.append(mbtn);

					mbtn.click(function(){
						$.post('/delete_placeness',
							{'image_shortcode': item.image_shortcode,
							'delete_tag': post_data.tags[0]},
							function(result){
								console.log(result);
							},
							'json');
						
						mdiv.remove();
					});
					$('#target-result').append(mdiv);
				});
			},
			'json'
		);
	});
});

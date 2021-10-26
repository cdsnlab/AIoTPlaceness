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

	$( "body" ).keypress(function( event ) {
		if ( $("input").is(':focus') ){
			return;
		}
		else {
			/*if ( event.which == 114 ) { // keycode for 'r'
				window.location.href = "/reset";
			}*/
			console.log(event.which);
			if ( event.which == 112 || event.which == 44) { // keycode for 'p'
				window.location.href = "/prev";
			}
			if ( event.which == 110 || event.which == 46) { // keycode for 'n'
				window.location.href = "/next";
			}
			/*if ( event.which == 13 ) { // keycode for 'c'
				$("#comment").focus();
			}*/
		}
	});

	/*$("#comment").keypress(function( event ) {
		if ( event.keyCode == 13 ) {
			$(this).removeClass('active');
			window.location.href = "/next";
		}
	});*/
	$("#target").click(function() {
		//alert("x: " + event.clientX + " - y: " + event.clientY);
		console.log(nStruct);
	});

	console.log(simple_chart_config);
	var chart = new Treant(simple_chart_config);
	$('.node').click(function(){
		var ask = false;
		console.log($(this).text());
		if('Activity' == $(this).text()) return;
		if (!$(this).hasClass('switch-on')){
			$(this).addClass('switch-on');
			ask = true;
		}
		else{
			$(this).removeClass('switch-on');
		}

		var post_data = {
			image_shortcode: shortcode,
			tags: [],
			image_url: image_url
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

		$.post('/save_placeness',
			post_data,
			function(result){
				console.log(result);
				alert(result);
			},
			'json'
		);

		console.log(post_data.tags);
		if(ask && confirm(post_data.tags + "\n" + "Check next?")){
			window.location.href = '/next';
		}
	});
});

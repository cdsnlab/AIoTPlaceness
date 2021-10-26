$(function(){
	$('#message-close').click(function(){
		$('#message').hide();
		console.log("hello");
	});
	$('#discard').click(function(){
		if(confirm("Discard Everyting?")){
			window.location.href = '/';
		}
	});
	$('#save').click(function(){
		var text_ontology = $('#ta_ontology').val();
		$.post('/save_ontology',
			{data: text_ontology},
			function(msg){
				alert(msg);
				window.location.reload();
			},
		);
	});
	$('#save_and_rtn').click(function(){
		var text_ontology = $('#ta_ontology').val();
		$.post('/save_ontology',
			{data: text_ontology},
			function(msg){
				if (msg == "Success"){
					window.location.href = '/template';
				}
				else{
					alert("Error in saving");
				}
			},
		);
	});

});

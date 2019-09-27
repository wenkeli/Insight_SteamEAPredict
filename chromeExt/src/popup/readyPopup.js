function load()
{
	chrome.storage.local.get("steamEA", 
			function(data)
			{
				chrome.extension.getBackgroundPage().console.log(data)
				var predictStr=data.steamEA[0]
				var predictPerc=data.steamEA[1]
				chrome.extension.getBackgroundPage().console.log(predictStr)
				chrome.extension.getBackgroundPage().console.log(predictPerc)
				predictPerc=predictPerc.toString()+"%"
				
				var newH=document.createElement("H1")
				var newT=document.createTextNode("completion probability: "+predictPerc)
				newH.appendChild(newT)
				document.body.appendChild(newH)
			});
}

document.addEventListener("DOMContentLoaded", 
		function()
		{
			load();
		});

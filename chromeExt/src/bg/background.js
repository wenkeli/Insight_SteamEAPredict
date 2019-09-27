function reset () 
{
	// set the icon to greyscale 
	chrome.browserAction.setIcon({path : "../../icons/waiting.png"});
	chrome.browserAction.setPopup({popup : "src/popup/waitingPopup.html"});
	// clean the local storage
	chrome.storage.local.clear(
			function ()
			{
				console.log("Events reset");
			});
}

reset();

//API server IP
var api_server = "http://127.0.0.1:8000/";

var urlRE = new RegExp("https://store.steampowered.com/app/([0-9]+)/.*")

// Add a listenser when DOM is loaded.
chrome.webNavigation.onDOMContentLoaded.addListener(
		function (details) 
		{
			var url = details.url;
			reset();
		
			// If en.wikipedia.org is nativaged.
			if (urlRE.test(url))
			{
				var appID = urlRE.exec(url)[1]
				console.log(appID)
		
				// URL for http requests
				var reqURL = api_server + "predict?appid=" + appID;
		
				// Send http requests
				fetch(reqURL)
				.then(r => r.text())
				.then(function(result) 
						{
							resultDict = JSON.parse(result);
							console.log(resultDict.predictCat)
							console.log(resultDict.successProb)
							// Store the fetched data into local memory for display
							chrome.storage.local.set(
//									{predCat: resultDict.predictCat, predProb: resultDict.successProb},
									{steamEA: [resultDict.predictCat, resultDict.successProb]},
									function() {
										console.log("have prediction");
										// Change to colored icon
										chrome.browserAction.setIcon({path : "../../icons/ready.png"});
										chrome.browserAction.setPopup({popup : "src/popup/readyPopup.html"});
										
									});
						});
			}
		});



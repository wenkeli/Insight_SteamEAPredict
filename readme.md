# GameSaver
## Purpose
This project predicts the probability of completion of Steam Early Access games to help players find games that are likely to finish. The project uses the first 90 days of game development updates and user feedback to make this prediction so that the players can make an informed decision early and maximize the impact of their feedback.

## Structure
- modelBuild: contains the code to acquire data from Steam API, building features, and fitting models
- apiServer: Django python server for accessing the steam API for getting game data and serving predictions
- chromeExt: Google Chrome extension that queries the apiServer when the user visits the website of a Steam Early Access game, and displays the predictions.

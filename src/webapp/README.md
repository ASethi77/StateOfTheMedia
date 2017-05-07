# State Of The Media Demo Webapp

This app highlights our ability to predict presidential approval ratings given articles for a day
and overall sentiments. The tool allows you to add different articles for different dates and
shows how our model updates its predictions of presidential approval ratings per day in
an online manner.

## Dependencies

Make sure you have `npm` set up on your machine.

## Installation

1. Clone this repo.
2. Within the `src/webapp` directory, run `npm install`.
3. Run `npm start`.
4. Open a browser to `localhost:8000`.

## Developing

The main logic for the webapp is located within `app/app.ts`. This includes
the Angular controller for managing articles and plot models.

To work on this project, run `tsc` within the `src/webapp` directory,
followed by `npm start`. After doing so, you should be able to modify
`app/app.ts` and `app/index.html` and get updates as soon as you refresh
your browser.

## Technologies used

* AngularJS
* Bootstrap
* Chart.js
* Typescript
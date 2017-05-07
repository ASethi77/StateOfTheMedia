var FacialExpression = (function () {
    function FacialExpression(expression, path) {
        this.expression = expression;
        this.path = path;
        this.expressionName = expression;
        this.imgPath = path;
    }
    return FacialExpression;
}());
var StateOfTheMediaController = (function () {
    function StateOfTheMediaController($scope) {
        this.possibleExpressions = {
            "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
            "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
            "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
            "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg")
        };
        this.currentExpression = this.possibleExpressions["Happy"];
        this.showExpression = true;
        this.hideExpression = function () {
            this.showExpression = false;
        };
        this.displayExpression = function () {
            this.showExpression = true;
        };
        this.setExpression = function (newExpression) {
            this.hideExpression();
            this.currentExpression = newExpression;
            this.displayExpression();
        };
        // do nothing yet
    }
    return StateOfTheMediaController;
}());
StateOfTheMediaController.AngularDependencies = ['$scope', StateOfTheMediaController];
var app = angular.module('StateOfTheMediaApp', []);
app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);

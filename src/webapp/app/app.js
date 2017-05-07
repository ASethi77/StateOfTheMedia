var FacialExpression = (function () {
    function FacialExpression(expression, path) {
        this.expression = expression;
        this.path = path;
        this.expressionName = expression;
        this.imgPath = path;
    }
    return FacialExpression;
}());
var Article = (function () {
    function Article(text, date) {
        this.text = text;
        this.date = date;
        this.articleText = text;
        this.articleDate = date;
    }
    return Article;
}());
var StateOfTheMediaController = (function () {
    function StateOfTheMediaController($scope, $http) {
        // TODO: We shouldn't have to manually set AngularJS properties/modules
        // as attributes of our instance ourselves; ideally they should be
        // injected similar to the todomvc typescript + angular example on GH
        // <see https://github.com/tastejs/todomvc/blob/gh-pages/examples/typescript-angular/js/controllers/TodoCtrl.ts">
        this.$scope = null;
        this.$http = null;
        this.articlesPerDay = {};
        this.possibleExpressions = {
            "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
            "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
            "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
            "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg")
        };
        this.currentExpression = null;
        this.showExpression = false;
        this.addArticle = function (content, date) {
            var article = new Article(content, date);
            var dateKey = date.toDateString();
            var articleList;
            if (dateKey in this.articlesPerDay) {
                articleList = this.articlesPerDay[date.toDateString()];
            }
            else {
                articleList = [];
            }
            articleList.push(article);
            this.$http({
                method: "POST",
                data: articleList,
                url: "/sentiments"
            }).then(function success(response) {
                console.log(response);
            }, function error(response) {
                console.log(response);
            });
        };
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
        this.$scope = $scope;
        this.$http = $http;
    }
    return StateOfTheMediaController;
}());
StateOfTheMediaController.AngularDependencies = ['$scope', '$http', StateOfTheMediaController];
StateOfTheMediaController.$inject = [
    '$scope',
    '$http'
];
var app = angular.module('StateOfTheMediaApp', [
    'ngMockE2E'
]);
app.run(['$httpBackend', function ($httpBackend) {
        var sentiment = 25;
        console.log("apprunner");
        $httpBackend.whenPOST("/sentiments").respond(function (method, url, data) {
            return [200, sentiment, {}];
        });
    }]);
app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);

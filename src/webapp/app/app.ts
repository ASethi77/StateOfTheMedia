class FacialExpression {
    imgPath: string;
    expressionName: string;

    constructor(public expression: string, public path: string) {
        this.expressionName = expression;
        this.imgPath = path;
    }
}

class Article {
    articleDate: Date;
    articleText: string;

    constructor(public text: string, public date: Date) {
        this.articleText = text;
        this.articleDate = date;
    }
}

class StateOfTheMediaController {
    static AngularDependencies = [ '$scope', '$http', StateOfTheMediaController ];

    public static $inject = [
        '$scope',
        '$http'
    ];

    // TODO: We shouldn't have to manually set AngularJS properties/modules
    // as attributes of our instance ourselves; ideally they should be
    // injected similar to the todomvc typescript + angular example on GH
    // <see https://github.com/tastejs/todomvc/blob/gh-pages/examples/typescript-angular/js/controllers/TodoCtrl.ts">
    public $scope: ng.IScope = null;
    public $http = null;

    constructor($scope: ng.IScope, $http: ng.IModule) {
        this.$scope = $scope;
        this.$http = $http;

        // default date for demo
        this.$scope.articleDate = new Date(2016, 8, 21);
    }

    private articlesPerDay:  { [day: string]: Article[] } = {};
    private sentimentPerDay: { [day: string]: number } = {};

    private possibleExpressions: { [name: string]: FacialExpression } = {
        "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
        "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
        "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
        "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg"),
        "Neutral": new FacialExpression("Neutral", "./img/Neutral.jpg")
    };

    public currentExpression: FacialExpression = this.possibleExpressions["Neutral"];
    public showExpression: boolean = true;

    public addArticle = function (content: string, date: Date)
    {
        var article: Article = new Article(content, date);
        var dateKey: string = date.toDateString();

        var articleList: Article[];
        if (dateKey in this.articlesPerDay) {
            articleList = this.articlesPerDay[date.toDateString()];
        } else {
            articleList = [];
        }
        articleList.push(article);

        this.updateSentimentForDate(date);
    };

    public updateSentimentForDate(day: Date)
    {
        var sentimentPerDay = this.sentimentPerDay;
        var dateKey = day.toDateString();
        var articleList: Article[] = this.articlesPerDay[dateKey];

        var controller = this;
        this.$http({
            method: "POST",
            data: angular.toJson(articleList),
            url: "/sentiments"
        }).then(function success(response) {
                var sentiment: number = angular.fromJson(response.data);
                sentimentPerDay[dateKey] = sentiment;
                controller.currentExpression = controller.sentimentToFacialExpression(sentiment);
           }, function error(response) {
                console.log(response);
        });
    };

    public sentimentToFacialExpression(sentiment: number) {
        if (sentiment >= 0.0 && sentiment < 0.20) {
            return this.possibleExpressions["Really Sad"];
        } else if (sentiment >= 0.20 && sentiment < 0.40) {
            return this.possibleExpressions["Sad"];
        } else if (sentiment >= 0.40 && sentiment < 0.60) {
            return this.possibleExpressions["Neutral"];
        } else if (sentiment >= 0.60 && sentiment < 0.80) {
            return this.possibleExpressions["Happy"];
        } else if (sentiment >= 0.80 && sentiment <= 1.0) {
            return this.possibleExpressions["Really Happy"];
        } else {
            console.error("Unable to determine appropriate expression for " + sentiment.toPrecision(3));
            return this.currentExpression;
        }
}

var app = angular.module('StateOfTheMediaApp', [
    'ngMockE2E'
]);

app.run(['$httpBackend', function ($httpBackend) {
    $httpBackend.whenPOST("/sentiments").respond(function (method, url, data) {
        var sentiment = Math.random();
        return [200, sentiment, {}];
    });
}]);

app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);
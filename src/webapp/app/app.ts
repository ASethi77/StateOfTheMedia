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
    public $http: ng.IModule = null;

    constructor($scope: ng.IScope, $http: ng.IModule) {
        this.$scope = $scope;
        this.$http = $http;
    }

    private articlesPerDay:  { [day: string]: number[] } = {};

    private possibleExpressions: { [name: string]: FacialExpression } = {
        "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
        "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
        "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
        "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg")
    };

    public currentExpression: FacialExpression = null;
    public showExpression: boolean = false;

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

    private hideExpression = function ()
    {
        this.showExpression = false;
    };

    private displayExpression = function ()
    {
        this.showExpression = true;
    };

    private setExpression = function (newExpression: FacialExpression)
    {
        this.hideExpression();
        this.currentExpression = newExpression;
        this.displayExpression();
    };
}

var app = angular.module('StateOfTheMediaApp', [
    'ngMockE2E'
]);

app.run(['$httpBackend', function ($httpBackend) {
    var sentiment: number = 25;
    console.log("apprunner");
    $httpBackend.whenPOST("/sentiments").respond(function (method, url, data) {
        return [200, sentiment, {}];
    });
}]);

app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);
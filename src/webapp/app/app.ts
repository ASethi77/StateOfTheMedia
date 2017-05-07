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
    static AngularDependencies = [ '$scope', '$http', '$injector', StateOfTheMediaController ];

    public static $inject = [
        '$scope',
        '$http',
        '$injector'
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
        console.log(content);
        console.log(date);
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

var app = angular.module('StateOfTheMediaApp', []);
app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);
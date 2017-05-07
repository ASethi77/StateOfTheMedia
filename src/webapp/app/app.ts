class FacialExpression {
    imgPath: string;
    expressionName: string;

    constructor(public expression: string, public path: string) {
        this.expressionName = expression;
        this.imgPath = path;
    }
}

class StateOfTheMediaController {
    static AngularDependencies = [ '$scope', StateOfTheMediaController ];

    constructor($scope: ng.IScope) {
        // do nothing yet
    }

    private possibleExpressions: { [name: string]: FacialExpression } = {
        "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
        "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
        "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
        "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg")
    };

    public currentExpression: FacialExpression = this.possibleExpressions["Happy"];
    public showExpression: boolean = true;

    public hideExpression = function ()
    {
        this.showExpression = false;
    };

    public displayExpression = function ()
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
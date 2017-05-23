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
    static AngularDependencies = [ '$scope', '$http', '$cookies', StateOfTheMediaController ];

    public static $inject = [
        '$scope',
        '$http'
    ];

    // TODO: We shouldn't have to manually set AngularJS properties/modules
    // as attributes of our instance ourselves; ideally they should be
    // injected similar to the todomvc typescript + angular example on GH
    // <see https://github.com/tastejs/todomvc/blob/gh-pages/examples/typescript-angular/js/controllers/TodoCtrl.ts>
    public $scope: ng.IScope = null;
    public $http = null;

    private readonly SESSION_KEY_NAME: string = "StateOfTheMediaSession";

    constructor($scope: ng.IScope, $http: ng.IModule, $cookies) {
        this.$scope = $scope;
        this.$http = $http;

        // default date for demo
        this.$scope.articleDate = new Date(2016, 8, 21);

        // check to see if our cookie exists in the browser already
        if (!$cookies.get(this.SESSION_KEY_NAME)) {
            // console.log(this.SESSION_KEY_NAME);

            // if it doesn't exist, register a new session id with
            var newSessionState = getNewSession();
            this.$http({
                method: "POST",
                data: angular.toJson({"id": newSessionState}),
                url: "http://localhost:5000/register"
            }).then(function success(response) {
                $cookies.put(this.SESSION_KEY_NAME, newSessionState)
               }, function error(response) {
                    console.log(response);
            });
        }
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
    public topicLabels: string[] = [];
    public topicStrengths: number[] = [];
    public approvalRatingData: number[] = [];
    public lineChartLabels: string[] = [];
    public lineChartOptions: { legend: { display: true } };

    public dateFormattingOptions = {  
        year: "numeric", month: "short",  
        day: "numeric"
    };

    public getApprovalRatings = function(date: Date) {
        let controller = this;
        let args = {"date": date.toDateString()};
        this.$http.get("http://localhost:5000/approvalRatings", {params: args})
            .then(function success(response) {
                var responseData = angular.fromJson(response.data);
                controller.approvalRatingData = responseData['approvalRatings'];
                controller.lineChartLabels = responseData['labels'];
           }, function error(response) {
                console.log(response);
        });
    };

    public addArticle = function (content: string, date: Date)
    {
        var article: Article = new Article(content, date);
        var dateKey: string = date.toDateString();

        var articleList: Article[];
        if (dateKey in this.articlesPerDay) {
            articleList = this.articlesPerDay[date.toDateString()];
        } else {
            articleList = [];
            this.articlesPerDay[date.toDateString()] = articleList;
        }
        articleList.push(article);

        this.updateSentimentForDate(date);
    };

    public updateSentimentForDate(day: Date)
    {
        var sentimentPerDay = this.sentimentPerDay;
        var dateKey = day.toDateString();
        var articleList: Article[] = this.articlesPerDay[dateKey];

        let controller = this;
        this.$http({
            method: "POST",
            data: angular.toJson(articleList),
            url: "/nlp"
        }).then(function success(response) {
                var responseData = angular.fromJson(response.data);
                var sentiment: number = responseData['sentiment'];
                sentimentPerDay[dateKey] = sentiment;
                controller.currentExpression = controller.sentimentToFacialExpression(sentiment);

                controller.topicLabels = responseData['topicLabels'];
                controller.topicStrengths = responseData['topicStrengths'];
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


}

var app = angular.module('StateOfTheMediaApp', [
    'chart.js',
    // 'ngMockE2E',
    'ngCookies'
]);

app.config(function (ChartJsProvider) {
    // Configure all charts
    ChartJsProvider.setOptions({
        global: {
            colors: ['#97BBCD', '#DCDCDC', '#F7464A', '#46BFBD', '#FDB45C', '#949FB1', '#4D5360']
        }

    });
    // Configure all doughnut charts
    ChartJsProvider.setOptions('doughnut', {
      cutoutPercentage: 60
    });
    ChartJsProvider.setOptions('bubble', {
      tooltips: { enabled: false }
    });
});

app.run( function ($httpBackend) {
    // $httpBackend.whenPOST("/nlp").respond(function (method, url, data) {
    //     // generate fake sentiment
    //     var sentiment = Math.random();
    //
    //     // generate fake topic strengths for the day
    //     const NUM_TOPICS: number = 5;
    //     const TOPIC_LABELS = [ 'economy', 'foreign relations', 'war', 'social issues', 'other' ];
    //     var topics : number[] = [];
    //     var sum: number = 0.0;
    //     for (var i = 0 ; i < NUM_TOPICS; i++) {
    //         var topicValue = Math.random();
    //         topics.push(topicValue);
    //         sum += topicValue;
    //     }
    //     for (var i = 0; i < NUM_TOPICS; i++) {
    //         topics[i] = topics[i] / sum * 100.0;
    //     }
    //
    //     // generate approval rating
    //     var approvalRating: number = Math.random() * 100.0;
    //
    //     return [
    //         // response status code
    //         200,
    //
    //         // response data (sentiment, topics, predicted approval ratings for day)
    //         {
    //             'sentiment': sentiment,
    //             'topicLabels': TOPIC_LABELS,
    //             'topicStrengths': topics,
    //             'approval': approvalRating
    //         },
    //
    //         // extra headers (I think?)
    //         {}
    //     ];
    // });
    //
    // $httpBackend.whenGET("/approvalRatings").respond(function(url) {
    //     let approvalRatings = [70, 75, 70, 68, 80, 70, 72, 50, 80, 90, 20];
    //
    //     return [
    //         200,
    //
    //         {
    //             'approvalRatings': approvalRatings
    //         },
    //
    //         {}
    //     ];
    // });

     // $httpBackend.whenPOST("/register").respond(function (method, url, data) {
     //     return [
     //         200,
     //         {},
     //         {}
     //     ];
     // });


});

app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);

function getNewSession() {
    return Math.floor(Math.random() * (Number.MAX_VALUE - Number.MIN_VALUE + 1)) + Number.MIN_VALUE;
}

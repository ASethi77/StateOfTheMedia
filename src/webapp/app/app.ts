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
    articleHeadline: string;
    selectedForAnalysis: boolean;

    constructor(public text: string, public date: Date, public headline="") {
        this.articleText = text;
        this.articleDate = date;
        this.articleHeadline = headline;
        this.selectedForAnalysis = false;
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
    private readonly API_HOST_URL: string = "http://localhost";
    private readonly API_PORT: number = 5000;
    private sessionState;

    constructor($scope: ng.IScope, $http: ng.IModule, $cookies) {
        this.$scope = $scope;
        this.$http = $http;

        // default date for demo
        this.$scope.articleDate = new Date(1998, 09, 21);

        // check to see if our cookie exists in the browser already
        if (!$cookies.get(this.SESSION_KEY_NAME)) {
            // console.log(this.SESSION_KEY_NAME);

            // if it doesn't exist, register a new session id with
            var newSessionState = getNewSession();

            var registrationUrl = this.API_HOST_URL + ':' + this.API_PORT + '/register';
            this.$http({
                method: "POST",
                data: angular.toJson({"id": newSessionState}),
                url: registrationUrl
            }).then(function success(response) {
                $cookies.put(this.SESSION_KEY_NAME, newSessionState);
                console.log(this.sessionState);
               }, function error(response) {
                    console.log(response);
            });
        }
        this.sessionState = newSessionState;

        this.$http({
            method: "POST",
            data: angular.toJson({"id": newSessionState}),
            url: registrationUrl
        }).then(function success(response) {
            $cookies.put(this.SESSION_KEY_NAME, newSessionState);
            console.log(this.sessionState);
           }, function error(response) {
                console.log(response);
        });
    }

    public currentDelta: number = -1;
    public articlesSelectedForAnalysisIndices: { [day: string]: number[] } = {};
    public articlesSelectedForAnalysis:  { [day: string]: Article[] } = {};
    public articlesPerDay: { [day: string]: Article[] } = {};
    private sentimentPerDay: { [day: string]: number } = {};

    private possibleExpressions: { [name: string]: FacialExpression } = {
        "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
        "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
        "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
        "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg"),
        "Neutral": new FacialExpression("Neutral", "./img/Neutral.jpg")
    };

    public currentExpression: { [day: string]: FacialExpression } = {};
    public showExpression: boolean = true;
    public topicLabels: { [day: string]: string[] } = {};
    public topicStrengths:{ [day: string]: number[] } = {};
    public approvalRatingPredicted: { [day: string]: number } = {};
    public approvalRatingsLabels: string[] = ["approval", "disapproval", "neutral"];
    public lineChartLabels: string[] = [];
    public lineChartSeries: string[] = ["Projection (week from today)"];
    public lineChartOptions: { legend: { display: true } };

    public totalArticles : number = 0;

    public dateFormattingOptions = {
        year: "numeric", month: "short",
        day: "numeric"
    };

    public addAllMahShit = function(date: Date) {
        var dateStr = date.toDateString();
        var articlesForDate = this.articlesPerDay[dateStr];
        if (articlesForDate === undefined) {
            return;
        }

        for (var i = 0; i < articlesForDate.length; i++) {
            this.addArticleIndex(i, date, false);
        }

        this.updateAnalysis(date);
        // alert("I'm adding all yo shit");
    }

    public removeAllMahShit = function(date: Date) {
        var dateStr = date.toDateString();
        var articlesForDate = this.articlesPerDay[dateStr];
        if (articlesForDate === undefined) {
            return;
        }

        for (var i = 0; i < articlesForDate.length; i++) {
            this.removeArticleIndex(i, date, false);
        }

        this.updateAnalysis(date);
        // alert("I'm deleting all yo shit");
    }

    public getArticlesForDate = function(date: Date)
    {
        var dateStr = date.toDateString();
        var articlesForDate = this.articlesPerDay[dateStr];
        if (articlesForDate !== undefined) {
            return;
        }

        let controller = this;
        console.log(date.getDate());
        var articleUrl = this.API_HOST_URL + ':' + this.API_PORT + '/news/' + 
                         date.getFullYear() + '/' + 
                         (date.getMonth() + 1) + '/' + 
                         date.getDate();
        this.$http({
            'method': 'GET',
            url: articleUrl
        }).then(function success(response) {
            let articlesRaw = angular.fromJson(response.data);
            var articleList = [];
            articlesRaw.forEach(function (article) {
                articleList.push(new Article(article.content, date, article.headline));
            });
            controller.articlesPerDay[dateStr] = articleList;
            controller.currentDelta = Math.ceil(articleList.length / 3);
        }, function error(response) {
            console.error("unable to get articles for selected date");
            console.log(response);
        });
    }

    public getArticleColumn = function (index: number, date: Date) {
        var articleList = this.articlesPerDay[date.toDateString()];
        if (articleList === undefined)
        {
            return [];
        }
        var articleSubset = articleList.slice(index * this.currentDelta, (index + 1) * this.currentDelta);
        return articleSubset;
    };

    public getApprovalRatings = function(date: Date) {
        let controller = this;
        let args = {"date": date.toDateString()};
        var approvalRatingsUrl = this.API_HOST_URL + ':' + this.API_PORT + '/approvalRatings';
        this.$http.get(approvalRatingsUrl, {params: args})
            .then(function success(response) {
                var responseData = angular.fromJson(response.data);
                controller.approvalRatingData = responseData['approvalRatings'];
                controller.lineChartLabels = responseData['labels'];
            }, function error(response) {
                console.log(response);
            });
    };

    public addArticleIndex = function (index: number, date: Date, update=true)
    {
        let dateStr = date.toDateString();
        var articleIndicesForDay = this.articlesSelectedForAnalysisIndices[dateStr];
        if (articleIndicesForDay === undefined)
        {
            articleIndicesForDay = [ index ];
            this.articlesSelectedForAnalysisIndices[dateStr] = articleIndicesForDay;
            if (update === true) {
                this.updateAnalysis(date);
            }
            this.articlesPerDay[dateStr][index].selectedForAnalysis = true;
        } else if (articleIndicesForDay.indexOf(index) === -1)
        {
            articleIndicesForDay.push(index);
            if (update === true) {
                this.updateAnalysis(date);
            }
            this.articlesPerDay[dateStr][index].selectedForAnalysis = true;
        }
    }

    public removeArticleIndex = function (index: number, date: Date, update=true)
    {
        let dateStr = date.toDateString();
        var articleIndicesForDay = this.articlesSelectedForAnalysisIndices[dateStr];
        if (articleIndicesForDay !== undefined)
        {
            var indexToRemove = articleIndicesForDay.indexOf(index);
            if (indexToRemove > -1)
            {
                articleIndicesForDay.splice(indexToRemove, 1);
                if (update === true) {
                    this.updateAnalysis(date);
                }
                this.articlesPerDay[dateStr][index].selectedForAnalysis = false;
            }
        }
    };

    private updateAnalysis = function (date: Date)
    {
        let dateStr = date.toDateString();
        console.log("updating analysis for day's news");
        var articlesToAnalyze = [];
        var articleIndices = this.articlesSelectedForAnalysisIndices[dateStr];
        if (articleIndices.length === 0)
        {
            this.approvalRatingPredicted[dateStr] = undefined;
            return;
        }
        for (var i = 0; i < articleIndices.length; i++) {
            articlesToAnalyze.push(this.articlesPerDay[dateStr][articleIndices[i]]);
        };
        this.articlesSelectedForAnalysis[dateStr] = articlesToAnalyze;
        this.fetchSentimentMeasurement(date);
        this.fetchTopicMeasurement(date);
        this.fetchApprovalRatingPredictions(date);
    };

    // public addArticle = function (content: string, date: Date)
    // {
    //     if ("undefined" === typeof content) {
    //         return;
    //     }

    //     var newArticle = new Article(content, date);
    //     var dateStr = date.toDateString();
    //     var articleSetForDate = this.articlesSelectedForAnalysis[dateStr];
    //     if (articleSetForDate === undefined)
    //     {
    //         articleSetForDate = [ newArticle ];
    //         this.articlesSelectedForAnalysis[dateStr] = articleSetForDate;
    //     } else {
    //         articleSetForDate.push(newArticle);
    //     }
    //     this.totalArticles++;
    // };

    public fetchSentimentMeasurement = function (date: Date) {
        var dateStr = date.toDateString();
        var articlesToAnalyze = this.articlesSelectedForAnalysis[dateStr];
        if (articlesToAnalyze === undefined)
        {
            return;
        }

        var getSentimentUrl = this.API_HOST_URL + ':' + this.API_PORT + '/model/sentimentForDay';

        var sentimentReqData = {
            'id': this.sessionState,
            'articles': articlesToAnalyze,
            'day': dateStr
        };
        var controller = this;
        this.$http({
            method: "POST",
            data: angular.toJson(sentimentReqData),
            url: getSentimentUrl
        }).then(function success(response) {
            console.log("done computing sentiment for articles");
            console.log(response);
            controller.currentExpression[dateStr] =
                    controller.sentimentToFacialExpression(response.data.sentiment);
        }, function error (response) {
            alert("error computing sentiment for articles");
        });
    };

    public fetchTopicMeasurement = function (date: Date) {
        var dateStr = date.toDateString();
        var articlesToAnalyze = this.articlesSelectedForAnalysis[dateStr];
        if (articlesToAnalyze === undefined)
        {
            return;
        }

        var getTopicUrl = this.API_HOST_URL + ':' + this.API_PORT + '/model/topicMixtureForDay';
        var indicesToAnalyze = [];

        var topicReqData = {
            'id': this.sessionState,
            'articles': articlesToAnalyze,
            'day': dateStr
        };

        var controller = this;
        this.$http({
            method: "POST",
            data: angular.toJson(topicReqData),
            url: getTopicUrl
        }).then(function success(response) {
            console.log("done computing topics for articles");
            console.log(response);
            controller.topicLabels[dateStr] = response.data.topicLabels;
            for (var i = 0; i < response.data.topicStrengths.length; i++) {
                response.data.topicStrengths[i] = Math.round(response.data.topicStrengths[i] * 1000.0) / 1000.0;
            }
            controller.topicStrengths[dateStr] = response.data.topicStrengths;
        }, function error(response) {
            alert("error computing topics for articlces");
        });
    }

    public fetchApprovalRatingPredictions = function (date)
    {
        var dateStr = date.toDateString();
        var articlesToAnalyze = this.articlesSelectedForAnalysis[dateStr];
        if (articlesToAnalyze === undefined)
        {
            return;
        }

        var apiReqData = {
            'id': this.sessionState,
            'articles': articlesToAnalyze,
            'day': dateStr
        };

        var predictionUrl = this.API_HOST_URL + ':' + this.API_PORT + '/model/predict';
        var controller = this;
        this.$http({
            method: "POST",
            data: angular.toJson(apiReqData),
            url: predictionUrl
        }).then(function success(response) {
            console.log("predicted something");
            console.log(response);
            var predictions = response.data.prediction;
            for (var i = 0; i < predictions.length; i++)
            {
                predictions[i] = Math.round(predictions[i] * 100.0) / 100.0;
            }
            controller.approvalRatingPredicted[dateStr] = [predictions];
            console.log(controller.approvalRatingPredicted[dateStr]);
        }, function error(response) {
            alert("error predicting approval rating");
            console.log(response);
        });
    };

    public sentimentToFacialExpression(sentiment: number) {
        if (sentiment >= -1.0 && sentiment < -0.60) {
            return this.possibleExpressions["Really Sad"];
        } else if (sentiment >= -0.60 && sentiment < -0.20) {
            return this.possibleExpressions["Sad"];
        } else if (sentiment >= -0.20 && sentiment < 0.20) {
            return this.possibleExpressions["Neutral"];
        } else if (sentiment >= 0.20 && sentiment < 0.60) {
            return this.possibleExpressions["Happy"];
        } else if (sentiment >= 0.60 && sentiment <= 1.0) {
            return this.possibleExpressions["Really Happy"];
        } else {
            console.error("Unable to determine appropriate expression for " + sentiment.toPrecision(3));
            return this.currentExpression;
        }
    }
}

var app = angular.module('StateOfTheMediaApp', [
    'chart.js',
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
    // Removed mock code.
});

app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);

function getNewSession() {
    return Math.floor(Math.random() * (Number.MAX_VALUE - Number.MIN_VALUE + 1)) + Number.MIN_VALUE;
}

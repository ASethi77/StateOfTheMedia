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
    function Article(text, date, headline) {
        if (headline === void 0) { headline = ""; }
        this.text = text;
        this.date = date;
        this.headline = headline;
        this.articleText = text;
        this.articleDate = date;
        this.articleHeadline = headline;
        this.selectedForAnalysis = false;
    }
    return Article;
}());
var StateOfTheMediaController = (function () {
    function StateOfTheMediaController($scope, $http, $cookies) {
        // TODO: We shouldn't have to manually set AngularJS properties/modules
        // as attributes of our instance ourselves; ideally they should be
        // injected similar to the todomvc typescript + angular example on GH
        // <see https://github.com/tastejs/todomvc/blob/gh-pages/examples/typescript-angular/js/controllers/TodoCtrl.ts>
        this.$scope = null;
        this.$http = null;
        this.SESSION_KEY_NAME = "StateOfTheMediaSession";
        this.API_HOST_URL = "http://localhost";
        this.API_PORT = 5000;
        this.currentDelta = -1;
        this.articlesSelectedForAnalysisIndices = {};
        this.articlesSelectedForAnalysis = {};
        this.articlesPerDay = {};
        this.sentimentPerDay = {};
        this.possibleExpressions = {
            "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
            "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
            "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
            "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg"),
            "Neutral": new FacialExpression("Neutral", "./img/Neutral.jpg")
        };
        this.currentExpression = {};
        this.showExpression = true;
        this.topicLabels = {};
        this.topicStrengths = {};
        this.approvalRatingPredicted = {};
        this.approvalRatingsLabels = ["approval", "disapproval", "neutral"];
        this.lineChartLabels = [];
        this.lineChartSeries = ["Projection (week from today)"];
        this.totalArticles = 0;
        this.dateFormattingOptions = {
            year: "numeric", month: "short",
            day: "numeric"
        };
        this.addAllMahShit = function (date) {
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
        };
        this.removeAllMahShit = function (date) {
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
        };
        this.getArticlesForDate = function (date) {
            var dateStr = date.toDateString();
            var articlesForDate = this.articlesPerDay[dateStr];
            if (articlesForDate !== undefined) {
                return;
            }
            var controller = this;
            console.log(date.getDate());
            var articleUrl = this.API_HOST_URL + ':' + this.API_PORT + '/news/' +
                date.getFullYear() + '/' +
                (date.getMonth() + 1) + '/' +
                date.getDate();
            this.$http({
                'method': 'GET',
                url: articleUrl
            }).then(function success(response) {
                var articlesRaw = angular.fromJson(response.data);
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
        };
        this.getArticleColumn = function (index, date) {
            var articleList = this.articlesPerDay[date.toDateString()];
            if (articleList === undefined) {
                return [];
            }
            var articleSubset = articleList.slice(index * this.currentDelta, (index + 1) * this.currentDelta);
            return articleSubset;
        };
        this.getApprovalRatings = function (date) {
            var controller = this;
            var args = { "date": date.toDateString() };
            var approvalRatingsUrl = this.API_HOST_URL + ':' + this.API_PORT + '/approvalRatings';
            this.$http.get(approvalRatingsUrl, { params: args })
                .then(function success(response) {
                var responseData = angular.fromJson(response.data);
                controller.approvalRatingData = responseData['approvalRatings'];
                controller.lineChartLabels = responseData['labels'];
            }, function error(response) {
                console.log(response);
            });
        };
        this.addArticleIndex = function (index, date, update) {
            if (update === void 0) { update = true; }
            var dateStr = date.toDateString();
            var articleIndicesForDay = this.articlesSelectedForAnalysisIndices[dateStr];
            if (articleIndicesForDay === undefined) {
                articleIndicesForDay = [index];
                this.articlesSelectedForAnalysisIndices[dateStr] = articleIndicesForDay;
                if (update === true) {
                    this.updateAnalysis(date);
                }
                this.articlesPerDay[dateStr][index].selectedForAnalysis = true;
            }
            else if (articleIndicesForDay.indexOf(index) === -1) {
                articleIndicesForDay.push(index);
                if (update === true) {
                    this.updateAnalysis(date);
                }
                this.articlesPerDay[dateStr][index].selectedForAnalysis = true;
            }
        };
        this.removeArticleIndex = function (index, date, update) {
            if (update === void 0) { update = true; }
            var dateStr = date.toDateString();
            var articleIndicesForDay = this.articlesSelectedForAnalysisIndices[dateStr];
            if (articleIndicesForDay !== undefined) {
                var indexToRemove = articleIndicesForDay.indexOf(index);
                if (indexToRemove > -1) {
                    articleIndicesForDay.splice(indexToRemove, 1);
                    if (update === true) {
                        this.updateAnalysis(date);
                    }
                    this.articlesPerDay[dateStr][index].selectedForAnalysis = false;
                }
            }
        };
        this.updateAnalysis = function (date) {
            var dateStr = date.toDateString();
            console.log("updating analysis for day's news");
            var articlesToAnalyze = [];
            var articleIndices = this.articlesSelectedForAnalysisIndices[dateStr];
            if (articleIndices.length === 0) {
                this.approvalRatingPredicted[dateStr] = undefined;
                return;
            }
            for (var i = 0; i < articleIndices.length; i++) {
                articlesToAnalyze.push(this.articlesPerDay[dateStr][articleIndices[i]]);
            }
            ;
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
        this.fetchSentimentMeasurement = function (date) {
            var dateStr = date.toDateString();
            var articlesToAnalyze = this.articlesSelectedForAnalysis[dateStr];
            if (articlesToAnalyze === undefined) {
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
            }, function error(response) {
                alert("error computing sentiment for articles");
            });
        };
        this.fetchTopicMeasurement = function (date) {
            var dateStr = date.toDateString();
            var articlesToAnalyze = this.articlesSelectedForAnalysis[dateStr];
            if (articlesToAnalyze === undefined) {
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
        };
        this.fetchApprovalRatingPredictions = function (date) {
            var dateStr = date.toDateString();
            var articlesToAnalyze = this.articlesSelectedForAnalysis[dateStr];
            if (articlesToAnalyze === undefined) {
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
                for (var i = 0; i < predictions.length; i++) {
                    predictions[i] = Math.round(predictions[i] * 100.0) / 100.0;
                }
                controller.approvalRatingPredicted[dateStr] = [predictions];
                console.log(controller.approvalRatingPredicted[dateStr]);
            }, function error(response) {
                alert("error predicting approval rating");
                console.log(response);
            });
        };
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
                data: angular.toJson({ "id": newSessionState }),
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
            data: angular.toJson({ "id": newSessionState }),
            url: registrationUrl
        }).then(function success(response) {
            $cookies.put(this.SESSION_KEY_NAME, newSessionState);
            console.log(this.sessionState);
        }, function error(response) {
            console.log(response);
        });
    }
    StateOfTheMediaController.prototype.sentimentToFacialExpression = function (sentiment) {
        if (sentiment >= -1.0 && sentiment < -0.60) {
            return this.possibleExpressions["Really Sad"];
        }
        else if (sentiment >= -0.60 && sentiment < -0.20) {
            return this.possibleExpressions["Sad"];
        }
        else if (sentiment >= -0.20 && sentiment < 0.20) {
            return this.possibleExpressions["Neutral"];
        }
        else if (sentiment >= 0.20 && sentiment < 0.60) {
            return this.possibleExpressions["Happy"];
        }
        else if (sentiment >= 0.60 && sentiment <= 1.0) {
            return this.possibleExpressions["Really Happy"];
        }
        else {
            console.error("Unable to determine appropriate expression for " + sentiment.toPrecision(3));
            return this.currentExpression;
        }
    };
    return StateOfTheMediaController;
}());
StateOfTheMediaController.AngularDependencies = ['$scope', '$http', '$cookies', StateOfTheMediaController];
StateOfTheMediaController.$inject = [
    '$scope',
    '$http'
];
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
app.run(function ($httpBackend) {
    // Removed mock code.
});
app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);
function getNewSession() {
    return Math.floor(Math.random() * (Number.MAX_VALUE - Number.MIN_VALUE + 1)) + Number.MIN_VALUE;
}

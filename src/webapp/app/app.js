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
        // <see https://github.com/tastejs/todomvc/blob/gh-pages/examples/typescript-angular/js/controllers/TodoCtrl.ts>
        this.$scope = null;
        this.$http = null;
        this.articlesPerDay = {};
        this.sentimentPerDay = {};
        this.possibleExpressions = {
            "Really Happy": new FacialExpression("Really Happy", "./img/ReallyHappy.jpg"),
            "Happy": new FacialExpression("Happy", "./img/Happy.jpg"),
            "Sad": new FacialExpression("Sad", "./img/Sad.jpg"),
            "Really Sad": new FacialExpression("Really Sad", "./img/ReallySad.jpg"),
            "Neutral": new FacialExpression("Neutral", "./img/Neutral.jpg")
        };
        this.currentExpression = this.possibleExpressions["Neutral"];
        this.showExpression = true;
        this.topicLabels = [];
        this.topicStrengths = [];
        this.dateFormattingOptions = {
            year: "numeric", month: "short",
            day: "numeric"
        };
        this.addArticle = function (content, date) {
            var article = new Article(content, date);
            var dateKey = date.toDateString();
            var articleList;
            if (dateKey in this.articlesPerDay) {
                articleList = this.articlesPerDay[date.toDateString()];
            }
            else {
                articleList = [];
                this.articlesPerDay[date.toDateString()] = articleList;
            }
            articleList.push(article);
            this.updateSentimentForDate(date);
        };
        this.$scope = $scope;
        this.$http = $http;
        // default date for demo
        this.$scope.articleDate = new Date(2016, 8, 21);
    }
    StateOfTheMediaController.prototype.updateSentimentForDate = function (day) {
        var sentimentPerDay = this.sentimentPerDay;
        var dateKey = day.toDateString();
        var articleList = this.articlesPerDay[dateKey];
        var controller = this;
        this.$http({
            method: "POST",
            data: angular.toJson(articleList),
            url: "/nlp"
        }).then(function success(response) {
            var responseData = angular.fromJson(response.data);
            var sentiment = responseData['sentiment'];
            sentimentPerDay[dateKey] = sentiment;
            controller.currentExpression = controller.sentimentToFacialExpression(sentiment);
            controller.topicLabels = responseData['topicLabels'];
            controller.topicStrengths = responseData['topicStrengths'];
        }, function error(response) {
            console.log(response);
        });
    };
    ;
    StateOfTheMediaController.prototype.sentimentToFacialExpression = function (sentiment) {
        if (sentiment >= 0.0 && sentiment < 0.20) {
            return this.possibleExpressions["Really Sad"];
        }
        else if (sentiment >= 0.20 && sentiment < 0.40) {
            return this.possibleExpressions["Sad"];
        }
        else if (sentiment >= 0.40 && sentiment < 0.60) {
            return this.possibleExpressions["Neutral"];
        }
        else if (sentiment >= 0.60 && sentiment < 0.80) {
            return this.possibleExpressions["Happy"];
        }
        else if (sentiment >= 0.80 && sentiment <= 1.0) {
            return this.possibleExpressions["Really Happy"];
        }
        else {
            console.error("Unable to determine appropriate expression for " + sentiment.toPrecision(3));
            return this.currentExpression;
        }
    };
    return StateOfTheMediaController;
}());
StateOfTheMediaController.AngularDependencies = ['$scope', '$http', StateOfTheMediaController];
StateOfTheMediaController.$inject = [
    '$scope',
    '$http'
];
var app = angular.module('StateOfTheMediaApp', [
    'chart.js',
    'ngMockE2E'
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
app.run(['$httpBackend', function ($httpBackend) {
        $httpBackend.whenPOST("/nlp").respond(function (method, url, data) {
            // generate fake sentiment
            var sentiment = Math.random();
            // generate fake topic strengths for the day
            var NUM_TOPICS = 5;
            var TOPIC_LABELS = ['economy', 'foreign relations', 'war', 'social issues', 'other'];
            var topics = [];
            var sum = 0.0;
            for (var i = 0; i < NUM_TOPICS; i++) {
                var topicValue = Math.random();
                topics.push(topicValue);
                sum += topicValue;
            }
            for (var i = 0; i < NUM_TOPICS; i++) {
                topics[i] = topics[i] / sum * 100.0;
            }
            // generate approval rating
            var approvalRating = Math.random() * 100.0;
            return [
                // response status code
                200,
                // response data (sentiment, topics, predicted approval ratings for day)
                {
                    'sentiment': sentiment,
                    'topicLabels': TOPIC_LABELS,
                    'topicStrengths': topics,
                    'approval': approvalRating
                },
                // extra headers (I think?)
                {}
            ];
        });
    }]);
app.controller('StateOfTheMediaController', StateOfTheMediaController.AngularDependencies);

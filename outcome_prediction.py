from os.path import exists

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def load_data(fname):
    # Make sure it exists
    if not exists(fname):
        raise Exception('USER ERROR: Couldnt find the file <%s>..' % fname)

    # Read the csv as a df
    df = pd.read_csv(fname)
    return df


def get_quarter_df(df, quarter, start=True):
    if start:
        minutes = (df.remainingMinutes >= 6)
    else:
        minutes = (df.remainingMinutes < 6)
    return df[
        (df.quarter == quarter) &
        minutes
        ].sort_values(by=['playId'])


def get_lead_changes(df):
    df['lead'] = df.apply(
        lambda x: 1 if x['homeScore'] > x['awayScore'] else (-1 if x['homeScore'] < x['awayScore'] else 0), axis=1)

    lead_dict = {}
    for gameId in df.gameId.unique():
        game_df = df[df.gameId == gameId]
        lead_dict[gameId] = sum(np.sign(game_df['lead']).diff().ne(0))
    lead_df = pd.DataFrame.from_dict(lead_dict, orient='index')
    lead_df.reset_index(inplace=True)
    lead_df.rename(columns={'index': 'gameId', 0: 'lead_changes'}, inplace=True)
    return lead_df


def get_score_stats(df):
    score_dict = {'gameId': [], 'homeScore': [], 'awayScore': []}
    for gameId in df.gameId.unique():
        game_df = df[df.gameId == gameId]
        last_row_df = game_df.iloc[-1:]
        score_dict['gameId'].append(last_row_df['gameId'].values[0])
        score_dict['homeScore'].append(last_row_df['homeScore'].values[0])
        score_dict['awayScore'].append(last_row_df['awayScore'].values[0])
    score_df = pd.DataFrame.from_dict(score_dict)
    score_df['pointDiff'] = score_df['homeScore'] - score_df['awayScore']
    score_df['pointSum'] = score_df['homeScore'] + score_df['awayScore']
    return score_df


def get_stats(df, quarter=None):
    stats_counts = df.groupby(['gameId', 'team', 'type']).size()
    stats_counts = stats_counts.reset_index()
    stats_counts.rename(columns={0: 'count'}, inplace=True)
    if quarter is not None:
        stats_counts['stat'] = 'q' + str(quarter) + '_' + stats_counts['team'] + '_' + stats_counts['type']
    else:
        stats_counts['stat'] = stats_counts['team'] + '_' + stats_counts['type']
    stats_counts.drop(columns=['team', 'type'], inplace=True)
    stats_df = stats_counts.pivot_table(index=['gameId'], columns=['stat'], values='count').fillna(0)
    stats_df.reset_index(inplace=True)

    lead_df = get_lead_changes(df)
    if quarter is not None:
        lead_df.rename(columns={
            'lead_changes': 'q' + quarter + '_lead_changes'
        }, inplace=True)
    stats_df = pd.merge(stats_df, lead_df, on='gameId')

    score_df = get_score_stats(df)
    if quarter is not None:
        score_df.rename(columns={
            'homeScore': 'q' + quarter + '_h_score',
            'awayScore': 'q' + quarter + '_a_score',
            'pointDiff': 'q' + quarter + '_pointDiff',
            'pointSum': 'q' + quarter + '_pointSum'
        }, inplace=True)
    stats_df = pd.merge(stats_df, score_df, on='gameId')

    return stats_df


def build_stats_df():
    temp_playbyplay_df = pd.merge(playbyplay_df, player_df[['gameId', 'playerId', 'teamId']], on=['gameId', 'playerId'])
    pbp_df = pd.merge(temp_playbyplay_df, game_df[['gameId', 'homeTeamId', 'awayTeamId']], on=['gameId'])
    pbp_df[['homeTeamId', 'awayTeamId']] = pbp_df[['homeTeamId', 'awayTeamId']].astype(int)
    pbp_df['team'] = np.where((pbp_df.teamId == pbp_df.homeTeamId), 'home', 'away')
    pbp_df.drop(columns=['teamId', 'homeTeamId', 'awayTeamId', 'playerId', 'time', 'dateTime'], inplace=True)
    # renaming type values
    pbp_df.type.replace(
        {
            'FieldGoalMissed': 'fg_missed', 'Rebound': 'reb', 'FieldGoalMade': 'fg_made', 'Steal': 'stl',
            'FreeThrowMissed': 'ft_missed', 'FreeThrowMade': 'ft_made',
            'Turnover': 'to', 'Goaltending': 'to', 'LaneViolation': 'to', 'Traveling': 'to', 'Palming': 'to',
            'ShootingFoul': 'pf', 'PersonalFoul': 'pf', 'OffensiveFoul': 'pf', 'LooseBallFoul': 'pf',
            'TechnicalFoul': 'pf', 'Foul': 'pf', 'FlagrantFoul': 'pf'
        },
        inplace=True
    )
    # dropping KickedBall and JumpBall
    pbp_df.drop(pbp_df[pbp_df.type.isin(['KickedBall', 'JumpBall'])].index, inplace=True)

    # q1_start_stats = get_stats(get_quarter_df(pbp_df, '1'), '1_s')
    # q1_end_stats = get_stats(get_quarter_df(pbp_df, '1', start=False), '1_e')
    # q2_start_stats = get_stats(get_quarter_df(pbp_df, '2'), '2_s')
    # q2_end_stats = get_stats(get_quarter_df(pbp_df, '2', start=False), '2_e')
    # # q3_start_stats = get_stats(get_quarter_df(pbp_df, '3'), '3_s')
    # # q3_end_stats = get_stats(get_quarter_df(pbp_df, '3', start=False), '3_e')
    # q1_df = pd.merge(q1_start_stats, q1_end_stats, on=['gameId'])
    # q2_df = pd.merge(q2_start_stats, q2_end_stats, on=['gameId'])
    # # q3_df = pd.merge(q3_start_stats, q3_end_stats, on=['gameId'])
    # stats_df = pd.merge(q1_df, q2_df, on=['gameId'])
    # # stats_df = pd.merge(stats_df, q3_df, on=['gameId'])
    # stats_df = pd.merge(stats_df, game_df[['gameId', 'pointsDiff']], on=['gameId'])

    game_quarter_array = pbp_df.groupby('gameId')['quarter'].unique()

    games_drop = []
    for index, quarter_array in game_quarter_array.items():
        if '1' not in quarter_array or '2' not in quarter_array:
            games_drop.append(index)

    pbp_df.drop(pbp_df[pbp_df.gameId.isin(games_drop)].index, inplace=True)
    # check without different stats for each quarter
    half_df = pbp_df[(pbp_df.quarter.isin(['1', '2']))].sort_values(by=['playId'])
    stats_df = get_stats(half_df)
    stats_df = pd.merge(stats_df, game_df[['gameId', 'pointsDiff']], on=['gameId'])

    stats_df.loc[stats_df['pointsDiff'] > 0, 'Result'] = 1
    stats_df.loc[(stats_df['pointsDiff'] < 0), 'Result'] = 0
    stats_df.loc[:, ~stats_df.columns.isin(['gameId'])] = stats_df.loc[:, ~stats_df.columns.isin(['gameId'])].astype(
        int)
    stats_df.drop(columns='pointsDiff', inplace=True)
    return stats_df


def experiments(X_train, y_train, X_test, y_test):
    dfs = []

    models = [
        ('LR', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC()),
        ('GNB', GaussianNB()),
        ('XGB', XGBClassifier())
    ]

    results = []

    names = []

    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

    target_names = ['win', 'loss']

    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))

        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)

    final = pd.concat(dfs, ignore_index=True)

    return final


def fine_tune_models(X_train, y_train, X_test, y_test):
    models = [
        ('LR', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC()),
        ('GNB', GaussianNB()),
        # ('XGB', XGBClassifier())
    ]

    params = [
        {
            'penalty': ['none', 'l1', 'l2'], 'C': [0.1, 1, 10]
        },
        {
            'bootstrap': [True], 'max_depth': [50, 100], 'max_features': [2, 3], 'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 5, 10], 'n_estimators': [100, 500, 1000]
        },
        {
            'n_neighbors': [3, 5, 7], 'p': [1, 2]
        },
        {
            'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly', 'sigmoid']
        },
        {
            'var_smoothing': np.logspace(0, -9, num=100)
            # },
            # {
            #     'max_depth': range(2, 10, 1),
            #     'n_estimators': range(60, 220, 40),
            #     'learning_rate': [0.1, 0.01, 0.05]
        }
    ]

    best_models = {}

    for (name, model), param in zip(models, params):
        print(name)
        best_model = fine_tune(model, param, X_train, y_train, X_test, y_test)

        best_models[name] = best_model

    return best_models


def fine_tune(model, params, X_train, y_train, X_test, y_test):
    target_names = ['win', 'loss']

    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    model_grid = GridSearchCV(estimator=model,
                              param_grid=params,
                              cv=kfold,
                              verbose=1,
                              scoring='accuracy', n_jobs=-1)

    model_grid.fit(X_train, y_train)
    best_model = model_grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=target_names))
    print('Accuracy: {}'.format(accuracy_score(y_pred, y_test.Result.values)))
    return best_model


if __name__ == '__main__':
    playbyplay_df = load_data('./data/playbyplay_data_dump.csv')
    game_df = load_data('./data/match_data_dump.csv')
    player_df = load_data('./data/player_data_dump.csv')

    stats_filename = './data/half_stats.csv'
    if exists(stats_filename):
        stats_df = load_data(stats_filename)
    else:
        stats_df = build_stats_df()
        stats_df.to_csv(stats_filename, index=False)

    X = stats_df.loc[:, ~stats_df.columns.isin(['gameId', 'Result'])]
    y = stats_df[['Result']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)

    # final = experiments(X_train, y_train, X_test, y_test)

    best_models = fine_tune_models(X_train, y_train, X_test, y_test)

    features = X.columns.values
    lr_best_model = best_models['LR']
    importance = lr_best_model.coef_[0]
    # summarize feature importance
    for i, v in zip(features, importance):
        print('Feature: {}, Score: {}'.format(i, v))
    # plot feature importance
    from matplotlib import pyplot

    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.title('Importance of features when using Logistic Regression')
    pyplot.xticks(features)
    pyplot.show()

    rf_best_model = best_models['RF']
    importance = rf_best_model.feature_importances_
    # summarize feature importance
    for i, v in zip(features, importance):
        print('Feature: {}, Score: {}'.format(i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.title('Importance of features when using Random Forest')
    pyplot.xticks(features)
    pyplot.show()

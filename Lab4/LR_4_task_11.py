import argparse
import json
import numpy as np
from compute_scores import pearson_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the movie recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser


def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue

        for item in dataset[user]:
            if item not in dataset[input_user] or dataset[input_user][item] == 0:
                overall_scores.update({item: dataset[user][item] * similarity_score})
                similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()])

    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
    return [movie for _, movie in movie_scores]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    with open('ratings.json', 'r') as f:

        data = json.loads(f.read())

    print(f"\nMovie recommendations for {user}:")
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(f"{i + 1}. {movie}")
import argparse
import json
import numpy as np
from compute_scores import pearson_score


def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')


    scores = np.array([[x, pearson_score(dataset, user, x)]
                       for x in dataset if x != user])

    scores_sorted = np.argsort(scores[:, 1])[::-1]

    top_users = scores_sorted[:num_users]
    return scores[top_users]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find users similar to the input user')
    parser.add_argument("--user", dest="user", required=True, help='Input user')
    args = parser.parse_args()
    user = args.user

    with open('ratings.json', 'r') as f:
        data = json.loads(f.read())

    print(f"\nUsers similar to {user}:\n")
    similar_users = find_similar_users(data, user, 3)

    print('User\t\t\tSimilarity score')
    print('-' * 41)
    for item in similar_users:
        print(f"{item[0]}\t\t{round(float(item[1]), 2)}")
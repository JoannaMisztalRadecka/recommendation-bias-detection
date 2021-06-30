# Bias Detection Tree for recommendation models

Most of the recommendation algorithms are tuned to optimize some global objective function. However, the distribution of error may differ dramatically among different combinations of attributes, and such algorithms may lead to propagating hidden data biases. Most approaches to fairness evaluation and disparity detection are based on analyzing a single dimension selected a-priori, such as a sensitive user attribute or a protected category of products.

This repository contains an implementation and experiments with Bias Detection Tree - a model-agnostic technique to automatically detect the combinations of user and item attributes correlated with unfair treatment by the recommender.

The proposed approach applies the CHAID decision tree to detect inequalities in error distribution for rating predictions.

The technical details and preliminary results were presented in a workshop paper:

Misztal-Radecka J., Indurkhya B. (2021) *When Is a Recommendation Model Wrong? A Model-Agnostic Tree-Based Approach to Detecting Biases in Recommendations.* In: Boratto L., Faralli S., Marras M., Stilo G. (eds) Advances in Bias and Fairness in Information Retrieval. BIAS 2021. Communications in Computer and Information Science, vol 1418. Springer, Cham. https://doi.org/10.1007/978-3-030-78818-6_9

Datasets used in the experiments should be uploaded to the folder `data/`. Preprocessing is implemented for the following folders that should be downloaded:

- `data/ml-1m` - MovieLens 1M dataset https://grouplens.org/datasets/movielens/1m/,
- `data/ml-100k` - MovieLens 100K dataset https://grouplens.org/datasets/movielens/100k/
- `data/book-crossing` - http://www2.informatik.uni-freiburg.de/~cziegler/BX/
- `data/market-bias` - Marketing Bias datasets https://github.com/MengtingWan/marketBias/tree/master/data
- `data/Deskdrop` - article sharing dataset https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop

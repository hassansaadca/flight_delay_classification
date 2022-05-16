# Machine Learning at Scale
One of our courses involves the construction of machine learning models for large scale data sets. Our final project involved leveraging PySpark to predict flight delays for domestic flights in the US. More specifically, our team attempted to see if a flight would be at least 15 minutes delayed, at least 2 hours before the scheduled departure time.


### The Datasets:

**Flight information from 2015 to 2019**
* 31,746,841 x 109 dataframe
* Features include those related to the following categories:
    * Time Period
    * Airline
    * Origin
    * Destination
    * Departure Delay Indicator

**Weather data from 2015 to 2019**
* 630,904,436 x 177 dataframe
* Features include those related to the following categories:
    * Wind observations
    * Precipitation
    * Visibility observation
    * Air temperature
    * Snow depth



### Process:
Our project consisted of several stages over 5 weeks:
1) We first started with an EDA phase in which we used subsets of the data to find correlation between weather and flight delays
2) We then worked on a process to join the dataframes, a process which required us to think about how to increase efficiency and speed up the process. The join ended up taking about 7 hours in total.
3) Feature Engineering. This involved extracting static features (e.g. the weather forecast 2 hours before the flight in question) and dynamic/ time-series features (e.g. the number of flights delayed on the same day, grouped by carrier, airport, plane number, etc.). We also attempted to build network-related features that showed how flight delays propagated from one airport to the next within each 24 hour window.
4) Building a pipeline to read in the data from our Azure storage blob, split the data into 5 folds for k-fold cross validation, apply the necessary feature transformations, and run grid search cross validation for hyperparameter tuning
5) Evaluating our model based on the optimal hyperparameters.

### Goal (of this repository):

In order to not give away answers and code to future students, I have not included the feature engineering notebook that we put together, nor notebooks that show our strategy for joining the two dataframes above.

The purpose of this repo is to demonstrate proficiency with PySpark. We were able to learn to use it within just a couple weeks, and we built several effective machine learning models for our ~30M row dataset.

Our pipeline development can be seen in the `Model Pipeline - Framework and Runs.ipynb` notebook, with the final pipeline test for our optimal hyperparameters (using an XGBoost Model) can be found in `Final Pipeline Test.ipynb`.

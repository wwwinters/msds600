# Winters Week 5 Assignment -- autoML
- Date: September 25, 2020
## Deliverables
- Jupyter Notebook exploring the tpot package
- An automation script that uses the model from the notebook to make predictions on a data set
## Summary
I started this exercise using the **pycaret** libraries and packages.  It took some time, but I was able to get it to work . . . somewhat.  After watching the lecture, I switched to **tpot**.  It appears to be more stable, but does not encapsulate the pipeline build to the same degree as **pycaret**.  Both techniques produced similar results on the new_churn_data set.  The results for both are [1 0 0 0 0] which do not match the stated answer of [1 0 0 1 0].  Not sure where the problem is and I will conduct a more thorough analysis in the future.

The quality of the training, test, and new_data is really important.  If I train a model using six features and the new data has seven, the prediction method will not work.  The method I used in the python script is really tailored to this exercise and on the job, I would take more time to build in error checking routines to make sure that the new data matches the training data.

Even with autoML processes, it can still take time to evaluate a set of models to find the best fit for the dataset.  With all of the work I've put on the **tpot** notebook, I still feel **pycaret** has more potential for easily selecting a modle and building a pipeline around it.  Both packages support GPU; therefore, I may enable that feature and see how much performance improvement my kind of old GPU can provide.

I noticed that `sklearn.prediction()` returns a *numpy.ndarray* which I wasn't sure how to handle.  I tried something like

```
for i in range(len(new_data)):
   print('Churn = '+(new_data[i], predictions[i])
```
but recieved a key error, so I just printed the array.
## Running the script
The script accepts two arguments `--bestmodel` and `datafile`
- **--bestmodel** has two choices, `etc` for *ExtraTreesClassifier()* and `xgbc` for *XGBClassifier()*
- **--datafile** This is the path to the file containing the dataset that predictions will be run on.
### Examples
```
cd msds600
./Winters_Week5_predict.py --datafile ../data/new_churn_data.csv --bestmodel xgbc
./Winters_Week5_predict.py --datafile ../data/new_churn_data.csv --bestmodel etc
```
### Concerns
- This is a MVP release and does not have robust error checking included
- The prediction provided by both models do not match the stated answer of [1,0,0,1,0]
- Only printed the *numpy* array
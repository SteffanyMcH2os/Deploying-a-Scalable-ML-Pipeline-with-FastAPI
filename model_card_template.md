# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was completed by Steffany McWaters for the Deploying a Scalable ML Pipeline with FastAPI Udacity course project. This is the initial project version.  
The model uses scikit-learn's GradientBoostingClassifier with default parameters and random_state = 42. It predicts whether a person's annual income falls into the '<=50K' or '>50K' category. No hyperparameter tuning was performed.
The input features include age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country.

## Intended Use

This model is intended for an educational demonstration of machine learning training, evaluation, and deployment through a FastAPI application. Its intended users are students, instructors, and developers reviewing the project.

## Training Data

The model was trained using the Census income dataset supplied as census.csv in the starter repository. Dataset documentation is available at https://archive.ics.uci.edu/dataset/20/census+income. The data was divided into training and testing sets using an 80/20 split with random_state=42. The salary column was used as the target and excluded from the input features. Categorical features were processed with OneHotEncoder, and salary labels were processed with LabelBinarizer. These preprocessing objects were fitted using the training data. Continuous features were not scaled. Missing entries represented by '?' were retained as categorical values. No imputation was performed.

## Evaluation Data

The test set contained 6,513 records. It was processed using the encoder and label binarizer fitted on the training data. Performance was also evaluated separately for categories within workclass, education, marital-status, occupation, relationship, race, sex, and native-country, and the results were saved in slice_output.txt.

## Metrics

The model was evaluated using precision, recall, and F1 for the '>50K' class. On the overall test set, precision was 0.8030, recall was 0.6225, and F1 was 0.7013. These results indicate that 80.30% of records predicted as '>50K' actually belonged to that category. The model identified 62.25% of all actual '>50K' records. 
Performance varied across slices. For the female category, precision was 0.7973, recall was 0.5064, and F1 was 0.6194 over 2,126 records. For males, precision was 0.8037, recall was 0.6428, and F1 was 0.7143 over 4,387 records. The HS-grad education category had an F1 of 0.4161 compared with 0.7721 for Bachelors. Full categorical slice results are recorded in slice_output.txt.
The supplied metric function uses zero_division=1. Undefined metrics are therefore reported as 1.000, which should not be interpreted as evidence of perfect performance. 

## Ethical Considerations

The dataset may reflect historical inequalities and biases associated with income. The model uses demographic features, including sex, race and age, and may reproduce societal patterns of disadvantage. Differences in performance across demographic slices warrant further investigation. Slice metrics alone do not establish that the model is fair or identify the causes of those differences. Predictions should not be used to make decisions about individuals.

## Caveats and Recommendations

Results are based on one train/test split of the supplied dataset and may not generalize to current populations or other settings. Retaining '?' as a category allows the model to process missing entries but does not recover the missing information. Some groups contain very few test records, so their results should be interpreted cautiously. Further testing with additional data and demographic groups is recommended before using the model outside of this educational project.
### Prerequisities
Install dependencies - `pip install -r requirements.txt`

Place data into `data` directory under names:
* `llm_dark_patterns.xls` - data with responses to feature-enriched prompts
* `baseline` - data with baseline responses with no features
### Raw text features analysis
To reproduce these results run notebooks from directory `text_feature_analysis`. Order does not matter.

### Manipulation-related text features analysis
Calculate manipulation features using following script:
``` bash
cd manipulation_feature_analysis

python calculate_manipulation_features_from_liwc.py --data ../data/baseline.xlsx -o ../data/processed

python calculate_manipulation_features_from_liwc.py --data ../data/llm_dark_patterns.xls -o ../data/processed
```

And then run notebooks. Order does not matter.
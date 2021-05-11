# recsys2021

## Step 0: Prepare the data
I used a sample toy data here: `data/part-00000-sample.csv`, which is created from the raw file `part-00000`.

Code snippet:
```python
all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

with open("data/part-00000", encoding="utf-8") as f:
    all_data = []
    for line in f.readlines()[:]:
        record = {}
        line = line.strip()
        features = line.split("\x01")
        record.update({feature: features[idx] for feature, idx in all_features_to_idx.items()})
        record.update({label: features[idx]  for label, idx in labels_to_idx.items()})
        all_data.append(record)

df_all_data = pd.DataFrame.from_dict(all_data)
df_all_data[:10000].to_csv("../data/part-00000.csv", mode="w", index=False)
```
Note: when I write the data sample into a `.csv` file, types for targets `['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']` are changed from "string" to "float" (I dunno how to keep the string format).

So inside the data loading in `commons.data_provision.batch_read_dask()`, I manually load these four columns as "str" again, but this leaves some "NaN" data in the final loaded data (the original data was some empty string '' data).

The `training.transformers.label_transform.LabelTransform()` class is designed specifically for this data format. One can always change this class if the data files or formats are different. Say, if we are reading from the original data files, we can then provide new data reading functions in `commons.data_provision` and provide new data label transformation methods in `training.transformers.label_transform`.

## Step 1: Set up `./config.json`
Note:
1. I only used one label here: "like_timestamp", since I dunno how to use multiple columns as labels inside LightGBM model training.
2. So current model is actually a naive binary classification model for one label.


## Step 2: Run `cloud_run.py`
```
python cloud_run.py
```

## run `sh verify.sh` before pushing
pylint + pytest (missing)

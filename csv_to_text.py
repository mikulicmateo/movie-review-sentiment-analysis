import os
import pandas as pd
import re, html

working_dir = os.getcwd()
positive_dir = os.path.join(working_dir, "Datasets/Updated_Dataset/pos")
negative_dir = os.path.join(working_dir, "Datasets/Updated_Dataset/neg")
df = pd.read_csv(os.path.join(working_dir, "Datasets/Updated_Dataset/IMDB Dataset.csv"))
positive_reviews = []
negative_reviews = []
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')


def write_reviews_to_txt(reviews, save_dir, file_name):
    os.chdir(save_dir)
    for i, review in enumerate(reviews):
        with open(f"{file_name}{i + 1}.txt", "w") as file:
            file.write(review)


def main():
    for _, row in df.iterrows():
        if row[1] == "positive":
            pruned_string = tag_re.sub('', row[0])
            positive_reviews.append(pruned_string.replace('\\', ''))
        else:
            pruned_string = tag_re.sub('', row[0])
            negative_reviews.append(pruned_string.replace('\\', ''))

    write_reviews_to_txt(positive_reviews, positive_dir, "pos_review")
    write_reviews_to_txt(negative_reviews, negative_dir, "neg_review")


if __name__ == "__main__":
    main()

import random
import string


def get_random_string(n):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def get_number_of_target_classes(df):
    targets = df["sirna"].tolist()
    return int(max(set(targets)) + 1)

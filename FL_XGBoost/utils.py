import pandas as pd
import numpy as np

def apply_age_skew(df, age_column, skew_range=(-0.55, 0.83)):
    """Apply skewing to age column."""
    skewed_data = df[(df[age_column] > skew_range[0]) & (df[age_column] <= skew_range[1])]
    return skewed_data

def apply_gender_skew(df, gender_column, male_fraction=0.8):
    """Apply skewing to gender column."""
    males = df[df[gender_column] == 1]
    females = df[df[gender_column] == 0]
    num_males = int(len(males) * male_fraction)
    num_females = len(females)
    skewed_data = pd.concat([males[:num_males], females])
    return skewed_data

def apply_education_skew(df, education_columns, cutoff="education_HS-grad"):
    """Apply skewing to education columns based on a cutoff level."""
    lower_education = df[df[cutoff] == 1]
    higher_education = df[df[cutoff] == 0]
    skewed_data = pd.concat([lower_education, higher_education])
    return skewed_data

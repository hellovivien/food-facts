import pandas as pd
import numpy as np
import ftfy
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from IPython.display import Markdown
from IPython.display import display
from IPython.display import HTML
import os




'''
PRINTING FUNCT
'''

def before(action, key, value):
    print(action)
    print(f"{key}: {value}")

def after(key, value):
    print("Done!")
    print(f"{key}: {value}")
    print("----------------------------------------------------------------------")

def table(fields = [], rows = []):
    x = PrettyTable()
    x.field_names = fields
    x.add_rows(rows)
    print(x)

def md(input):
    display(Markdown(input))
    
def line(text):
    print("----------------------------------------------------------------------")
    md(f"##### **{text}**")
    print("----------------------------------------------------------------------")

'''
COMMON FUNCT
'''

def get_per(total, value):
#     total = float(total)
#     value = float(value)
    if total > 0:
        return f"{round((value / total)*100, 2)}%"
    return "0%"

def get_value_with_perc(total, value):
    return f"{value}/{total} ({get_per(total, value)})"


def empty_string(value, empty_values = ()):
    if empty_number(value) or value in empty_values:
        return True
    value = str(value).lower()
    return value.isspace() or value == '""' or value == "''" or value == "0" or value == "null" or value == "none" or value == "nan" or value == "unknown"

def empty_number(value, zero_is_empty = False):
    return pd.isna(value) or value is None or (value == 0 and zero_is_empty)

def is_empty(value, zero_is_empty = False):
    if type(value) == "string":
        return empty_string(value)
    else:
        return empty_number(value)


'''
APPLY FUNCTIONS
'''


def apply_fix_encoding(value):
    if not is_empty(value):
        fixed = ftfy.fix_text(value, uncurl_quotes = False)
        if fixed != value:
            with open("fix_encoding.txt", "a") as fichier:
                fichier.write(f"-------------------------\nAVANT : {value}\nAPRES : {fixed}")
            return fixed
    return value

def is_valid_calc(row):
    fat = row["fat_100g"]
    sugars = row["sugars_100g"]
    proteins = row["proteins_100g"]
    carbs = row["carbohydrates_100g"]
    energy = row["energy_100g"]
    ignore_cols = ["nutrition-score-fr_100g", "nutrition-score-uk_100g", "energy_100g", "energy-from-fat_100g"]
    if not is_empty(fat) and not is_empty(carbs) and not is_empty(proteins) and not is_empty(energy):
        estimation = carbs*17 + proteins*17 + fat*38
        diff = abs(estimation - energy)
        if diff > 100:
            pass
            # print(diff, row.product_name, energy, fat, carbs, proteins, sep = " / ")
    if is_empty(carbs):
        carbs = sugars
    total_for_100g = fat + carbs + proteins
    if total_for_100g > 100:
        # le total des composants pour 100g est supérieur à 100g c'est refusé
        return False
    return True

def fix_sodium(row):
    if is_empty(row["sodium_100g"]) and not is_empty(row["salt_100g"]):
        print("fix sodium")
        return 0.4*row["salt_100g"]

def apply_copy_value_from_child(row, parent_col, child_cols):
    if is_empty(row[parent_col]):
        for col in child_cols:
            if not is_empty(row[col]):
                # debug
                #print(f"copie de valeur {col} -> {parent_col}")
                return row[col]
    return row[parent_col]

def is_french(row):
    france_cols = ["origins", "origins_tags", "origins_tags", "manufacturing_places", "manufacturing_places_tags",
                 "cities_tags", "purchase_places", "countries", "countries_tags", "countries_en"]
    for col in france_cols:
        value = str(row[col]).lower()
        if "fr" in value or "france" in value:
            return True  
    # return "fr-" in str(row["emb_codes_tags"]).lower() or "fr " in str(row["emb_codes"]).lower() or empty_string(str(row["emb_codes"]))
    return False


'''
DATAFRAME CLASS
'''

class Table(pd.DataFrame):

    def __init__(self, df):
        super().__init__(df)

    def print_cols(self):
        table(["name", "type"], [[col, self[col].dtype] for col in self])


    def get_cols(self):
        return self.columns.to_list()

    def remove_cols(self, cols):
        print(f"colonnes : {self.col_counts()}")
        self.drop(cols, axis = 1, inplace = True)
        print(f"Done. colonnes : {self.col_counts()}")


    def delete_outliers(self, col, mask):
        before = mask.sum()
        self[mask] = np.nan
        after = 0
        diff = before-after
        if diff > 0:
            print(f"supression de {diff} valeurs abérrantes sur la colonne {col}") 

    def fix_composition_values(self):
        for col in self:
            if col[-4:] == "100g":
                if col in ["nutrition-score-fr_100g", "nutrition-score-uk_100g"]:
                    self.delete_outliers(col, ((self[col]<-15) | (self[col]>40)))
                elif col in ["energy_100g"]:
                    self.delete_outliers(col, ((self[col]>7200) | (self[col]<0)))
                elif col in ["energy-from-fat_100g"]:
                    self.delete_outliers(col, ((self[col]>3800) | (self[col]<0)))
                elif col in ["carbon-footprint_100g"]:
                    self.delete_outliers(col, self[col]<0)
                else:
                    self.delete_outliers(col, ((self[col]>100) | (self[col]<0)))

        self["is_valid_calc"] = self.apply(is_valid_calc, axis = 1)

    
    def empty_to_nan(self):
        print("empty_to_nan")
        print("nan values : ", self.isna().sum().sum())
        empty_values = (None, " ", '""', "''", "null", "none", "nan")
        self.replace(empty_values, np.nan, inplace = True)
        print("nan values : ", self.isna().sum().sum())

    def clear_prod(self):
        cols = ['code', 'url', 'creator', 'created_t', 'created_datetime', 'last_modified_t', 'last_modified_datetime', 'image_url', 'image_small_url']
        self.remove_cols(cols)
        self.remove_duplicated_rows()
        self.remove_empty_cols()



    def clear_test(self):
        
        # self.print_cols()
        self.fix_encoding()
        self.empty_to_nan()
        self.copy_value_from_child()
        self.fix_composition_values()
        self["sodium_100g"] = self.apply(fix_sodium, axis = 1)
        self["is_french"] = self.apply(is_french, axis = 1)
        print("is_french :", self["is_french"].sum())
        self.drop(self[self.is_french == False].index, inplace = True) 
        # profile = ProfileReport(self, title='Pandas Profiling Report')
        # profile.to_file("your_report.html")
        # sns.boxplot(x=self["energy_100g"])
        # plt.savefig('save_as_a_png.png')
        # pd.DataFrame(self).to_csv("data_clean.csv", index = False)



    def remove_empty_cols(self):
        empty_cols = []
        for col_name in self:
            if self[col_name].size == self[col_name].apply(is_empty, args = (True, )).sum():
                empty_cols.append(col_name)
        if empty_cols:
            print(empty_cols)
            self.remove_cols(empty_cols)

    def copy_value_from_child(self):
        col_relations = {
            "product_name": ["generic_name"],
            "categories_tags": ["categories", "categories_en"],
            "main_category": ["main_category_en"],
            "traces_tags": ["traces", "traces_en"],
            "fruits-vegetables-nuts_100g": ["fruits-vegetables-nuts-estimate_100g"]
        }
        for parent_col, child_cols in col_relations.items():
            self[parent_col] = self.apply(apply_copy_value_from_child, args = (parent_col, child_cols,), axis = 1)


    def fix_encoding(self):
        os.system(f"rm fix_encoding")
        for col in self:
            if self[col].dtype == "object":
                self[col].apply(apply_fix_encoding)
        # value = str(value)
        # fix_value = ftfy.fix_text(value)
        # if fix_value != value:
        #     return fix_value
        # return False

    def row_counts(self):
        return self.shape[0]

    def col_counts(self):
        return self.shape[1]

    def remove_duplicated_rows(self):
        mask = self.duplicated()
        count = mask.sum()
        print(f"{count}/{self.row_counts()} lignes dupliquées.")
        self.drop_duplicates(inplace = True)
        print(f"Done. {self.row_counts()} lignes maintenant")


import argparse
import os
import pathlib
import re

import nltk
import pandas as pd
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("cmudict")
nltk.download("punkt_tab")


class LinguisticFeatureProcessor:
    def __init__(self, data_path: os.PathLike, output_path: os.PathLike):
        self.data_path = data_path
        self.output_path = output_path
        self.data = pd.ExcelFile(self.data_path)
        self.syllable_dict = cmudict.dict()

    def __del__(self):
        self.data.close()

    def get_sheet_names(self) -> list[str]:
        return self.data.sheet_names

    def _count_polysyllables(self, text: str) -> int:
        ctx: int = 0
        for word in word_tokenize(text):
            if word not in self.syllable_dict:
                continue
            if (
                max(
                    [
                        len([phoneme for phoneme in pronunciation if phoneme[-1].isdigit()])
                        for pronunciation in self.syllable_dict[word]
                    ]
                )
                >= 3
            ):
                ctx += 1
        return ctx

    def _count_normalized_unique_words(self, text: str) -> float:
        words = len(re.findall(r"\b\w+\b", text.lower()))
        unique_words = len(set(re.findall(r"\b\w+\b", text.lower())))
        return unique_words / words if words != 0 else 1.0
    
    def _count_unique_words(self, text: str) -> float:
        unique_words = len(set(re.findall(r"\b\w+\b", text.lower())))
        return unique_words

    def process_sheet(self, sheet_name: str) -> pd.DataFrame:
        sheet_data = pd.read_excel(self.data, sheet_name)

        sheet_data["SENT"] = sheet_data["response"].apply(lambda x: len(sent_tokenize(x)))

        sheet_data["polysyllables"] = sheet_data["response"].apply(
            lambda x: self._count_polysyllables(x)
        )
        # sheet_data["abstraction"] = (
        #     sheet_data["DAV"]
        #     + 2 * sheet_data["IAV"]
        #     + 3 * sheet_data["SV"]
        #     + 4 * sheet_data["adj"]
        # ) / (
        #     sheet_data["DAV"] + sheet_data["IAV"] + sheet_data["SV"] + sheet_data["adj"]
        # )
        if "unique_words_cnt" not in sheet_data.columns:
            sheet_data["unique_words_cnt"] = sheet_data["response"].apply(
                lambda x: self._count_normalized_unique_words(x)
            )
        sheet_data["unnormalized_unique_words_cnt"] = sheet_data["response"].apply(
            lambda x: self._count_unique_words(x)
        )
        
        sheet_data["lexical diversity"] = sheet_data["unique_words_cnt"]
        sheet_data["reading difficulty"] = (
            1.043 * sheet_data["polysyllables"].pow(1.0 / 2) * 30 / sheet_data["SENT"] + 3.1291
        )
        if "analytic" in sheet_data.columns:
            sheet_data["analytical"] = sheet_data["analytic"]
        else:
            sheet_data["analytical"] = sheet_data["Analytic"]
        sheet_data["self references"] = sheet_data["ipron"]
        sheet_data["certainty"] = sheet_data["certitude"]
        sheet_data["emotionality"] = sheet_data["emotion"]
        # sheet_data["hedges"] = sheet_data["Hedge"]
        return sheet_data

    def process_and_save(self, sheet_name: str) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.process_sheet(sheet_name).to_csv(self.output_path / (sheet_name + ".csv"))


def main():
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=pathlib.Path)
    parser.add_argument("-s", "--sheets", type=str, nargs="*")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(os.path.dirname(os.getcwd())) / "processed",
    )
    args = parser.parse_args()

    # validate arguments
    if (
        not args.data
        or not args.data.exists()
        or args.data.is_dir()
        or args.data.suffix not in (".xlsx", ".xls")
    ):
        raise RuntimeError("Please provie a path to raw liwc data in excel format!")

    linguistic_feature_processor = LinguisticFeatureProcessor(args.data, args.output)
    sheets = args.sheets or linguistic_feature_processor.get_sheet_names()
    for sheet in sheets:
        if sheet == "source":
            continue
        try:
            linguistic_feature_processor.process_and_save(sheet)
        except Exception as e:
            print(e)
            print(f"{sheet} is broken")


if __name__ == "__main__":
    main()

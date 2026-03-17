import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_properties_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    return df


def load_properties(city: str, csv_path: str) -> pd.DataFrame:

    data_source = os.getenv("DATA_SOURCE", "csv").lower()

    if data_source == "attom":
        # Future integration point:
        # call services.attom_client here and normalize response
        raise NotImplementedError("ATTOM integration scaffold exists, but is not enabled yet.")

    df = load_properties_from_csv(csv_path)

    if "CITY" in df.columns:
        return df[df["CITY"].astype(str).str.strip().str.lower() == city.lower()].copy()

    return df.copy()
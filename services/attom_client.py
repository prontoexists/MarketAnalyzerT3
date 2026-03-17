import os
import requests
from dotenv import load_dotenv

load_dotenv()

ATTOM_API_KEY = os.getenv("ATTOM_API_KEY")
ATTOM_BASE_URL = "https://api.gateway.attomdata.com"


def get_attom_property_detail(address1: str, address2: str):
    """
    Example ATTOM call scaffold.
    Not used yet unless DATA_SOURCE=attom and ATTOM_API_KEY is set.
    """
    if not ATTOM_API_KEY:
        raise ValueError("ATTOM_API_KEY is not set")

    url = f"{ATTOM_BASE_URL}/propertyapi/v1.0.0/property/detail"
    headers = {
        "accept": "application/json",
        "apikey": ATTOM_API_KEY,
    }
    params = {
        "address1": address1,
        "address2": address2,
    }

    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return response.json()
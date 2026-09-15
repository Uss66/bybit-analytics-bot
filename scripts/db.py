"""Shared DB connection helper for the ByBit analytics project."""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "bybit"),
        password=os.getenv("POSTGRES_PASSWORD", "change_me_local_dev"),
        dbname=os.getenv("POSTGRES_DB", "bybit_analytics"),
    )

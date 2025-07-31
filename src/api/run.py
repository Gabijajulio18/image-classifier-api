import uvicorn


def main() -> None:
    """Run the Flower classifier API using Uvicorn."""
    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000
    )  # python3 -m src.api.run to run


if __name__ == "__main__":
    main()

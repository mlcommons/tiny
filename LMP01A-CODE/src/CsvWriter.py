import datetime
import os


class CsvWriter:
    CSV_LOGS_FOLDER = "lpm01a_csv_files"

    def __init__(self, filename: str = None) -> None:
        """Initializes the CsvWriter with the given filename.

        Args:
            filename (str): The filename for the CSV file. If None, a filename will be generated based on the current date and time.
        """

        if filename is None:
            filename = self._make_filename()
        self._make_folder()

        print("Creating file: ", filename)

        self.filename = filename
        self.file = open(f"{self.CSV_LOGS_FOLDER}/{filename}", "w")

    def _make_filename(self) -> str:
        """Creates a filename based on the current time."""
        now = datetime.datetime.now()
        return f"lpm01a_{now.strftime('%H%M%S_%d%m%Y')}.csv"

    def _make_folder(self) -> None:
        """Creates the folder for the CSV files if it doesn't exist."""
        if not os.path.exists(self.CSV_LOGS_FOLDER):
            print("Creating folder: ", self.CSV_LOGS_FOLDER)
            os.makedirs(self.CSV_LOGS_FOLDER)

    def write(self, data: str) -> None:
        """Writes the given data to the file."""
        self.file.write(data)

    def close(self) -> None:
        """Closes the file."""
        print("Closing file: ", self.filename)
        self.file.close()

import os
import csv
import glob
import pandas as pd

class DataLoader():

    # List CSV Files In folder
    def filesInFolderWithPattern(self, folderPath, pattern):
        files = glob.glob(os.path.join(folderPath, pattern))
        return files

    # Load Data from CSV File
    def loadCSVFile(self, filePath):
        with open(filePath) as file:
            data = pd.read_csv(file, low_memory=False)
        return data

    # Load all CSV files in Folder
    def loadFilesInFolderWithPattern(self, folderPath, pattern):
        files = self.filesInFolderWithPattern(folderPath, pattern)
        frames = [self.loadCSVFile(file) for file in files]
        return pd.concat(frames, ignore_index=True)
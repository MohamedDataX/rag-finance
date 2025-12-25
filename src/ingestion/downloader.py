import os
from sec_edgar_downloader import Downloader
from datetime import datetime

class SECDownloader:
    def __init__(self, data_dir="data/raw", email="votre_email@example.com", company="Personnel"):

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.dl = Downloader(company, email, self.data_dir)

    def download_10k(self, ticker, limit=5):
 
        print(f"Démarrage du téléchargement des 10-K pour {ticker}...")
        try:
            # Télécharge les n derniers rapports 10-K
            count = self.dl.get("10-K", ticker, limit=limit)
            print(f"{count} rapports téléchargés pour {ticker} dans {self.data_dir}")
        except Exception as e:
            print(f"Erreur lors du téléchargement pour {ticker}: {e}")

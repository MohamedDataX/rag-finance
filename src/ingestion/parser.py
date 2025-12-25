import os
import re
from bs4 import BeautifulSoup
import glob

class SECParser:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = raw_dir

    def extract_mda(self, html_content):

        soup = BeautifulSoup(html_content, 'lxml')
        text = soup.get_text(separator=' ', strip=True)

        mda_pattern = re.compile(
            r"Item\s+7\.?\s+Management.?s\s+Discussion\s+and\s+Analysis.*?(?=Item\s+(?:7A|8)\.?)", 
            re.IGNORECASE | re.DOTALL
        )
        
        match = mda_pattern.search(text)
        
        if match:
            return match.group(0).strip()
        else:
            # Fallback : Si le regex échoue, on retourne un message (ou all txt)
            return None

    def process_files(self, ticker):

        search_path = os.path.join(self.raw_dir, "sec-edgar-filings", ticker, "10-K", "*", "*.txt")
        files = glob.glob(search_path)
        
        results = []
        
        print(f"Traitement de {len(files)} fichiers pour {ticker}...")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                mda_text = self.extract_mda(content)
                
                if mda_text:

                    result = {
                        "ticker": ticker,
                        "file_path": file_path,
                        "mda_content": mda_text[:500] + "...", # Apercu du contenu
                        "mda_length": len(mda_text)
                    }
                    results.append(result)
                    print(f" MD&A extrait : {len(mda_text)} caractères.")
                else:
                    print(f" Section MD&A non trouvée dans {os.path.basename(os.path.dirname(file_path))}")
                    
            except Exception as e:
                print(f" Erreur de lecture {file_path}: {e}")
                
        return results

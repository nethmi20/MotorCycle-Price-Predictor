from curl_cffi import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import random
from datetime import datetime

class IkmanBikeScraper:
    def __init__(self):
        self.base_url = "https://ikman.lk"
        self.search_url = f"{self.base_url}/en/ads/sri-lanka/motorbikes"
        self.vehicles = []
        
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://ikman.lk/"
        }

    def scrape_detail_page(self, url):
        details = {
            'title': 'Unknown',
            'make': 'Unknown',
            'model': 'Unknown',
            'yom': None,
            'mileage': None,
            'engine_cc': None,
            'price': None,
            'location': 'Unknown',
            'url': url,
            'scrape_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(url, headers=self.headers, impersonate="chrome120")
            if response.status_code != 200:
                print(f"      ⚠️ Detail page HTTP {response.status_code}")
                return details
            
            if "Just a moment" in response.text or "Cloudflare" in response.text or "Verify you are human" in response.text:
                print(f"      🛑 BLOCKED: Ikman served a captcha/bot challenge.")
                return details

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 1. Extract Title
            h1 = soup.find('h1')
            if h1:
                details['title'] = h1.get_text(strip=True)
                
            # 2. Extract Price
            full_text = soup.get_text()
            price_match = re.search(r'Rs\.?\s*([0-9,]+)', full_text)
            if price_match:
                clean_price = price_match.group(1).replace(',', '').strip()
                if clean_price:
                    try:
                        details['price'] = int(clean_price)
                    except ValueError:
                        pass

            # 3. Extract Location
            subtitle_divs = soup.find_all('div', class_=re.compile(r'subtitle'))
            for div in subtitle_divs:
                text = div.get_text(strip=True)
                if ',' in text:
                    raw_location = text.split(',')[-1].strip()
                    # Clean trailing junk like "3869views", "MEMBER", digits
                    clean_location = re.sub(r'[\d]+|views|members|member|MEMBER', '', raw_location, flags=re.IGNORECASE).strip()
                    if clean_location:
                        details['location'] = clean_location
                    break

            # 4. Extract Specs from label divs (class contains 'label--')
            specs_mapping = {
                'Brand:': 'make',
                'Model:': 'model',
                'Year of Manufacture:': 'yom',
                'Mileage:': 'mileage',
                'Engine capacity:': 'engine_cc'
            }
            
            # Find all label divs by their CSS class pattern
            label_divs = soup.find_all('div', class_=re.compile(r'label--'))
            
            for label_div in label_divs:
                label_text = label_div.get_text(strip=True)
                
                for label, key in specs_mapping.items():
                    if label_text == label:
                        value_node = label_div.find_next_sibling()
                        if value_node:
                            val = value_node.get_text(strip=True)
                            
                            if key == 'yom':
                                try: details['yom'] = int(val)
                                except ValueError: pass
                            elif key == 'mileage':
                                clean_val = val.replace('km', '').replace(',', '').strip()
                                try: details['mileage'] = int(clean_val)
                                except ValueError: pass
                            elif key == 'engine_cc':
                                clean_val = val.replace('cc', '').replace(',', '').strip()
                                try: details['engine_cc'] = int(clean_val)
                                except ValueError: pass
                            else:
                                details[key] = val
                        break

        except Exception as e:
            print(f"      ❌ Error scraping detail page: {e}")
        
        return details
    
    def scrape_page(self, page=1):
        url = self.search_url if page == 1 else f"{self.search_url}?page={page}"
        print(f"\n📄 Scraping Ikman page {page}: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, impersonate="chrome120")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/en/ad/' in href and href not in links:
                        links.append(href)
                
                print(f"   Found {len(links)} unique bike listings")
                
                new_vehicles = 0
                for idx, href in enumerate(links):
                    full_url = self.base_url + href if not href.startswith('http') else href
                    print(f"   [{idx+1}/{len(links)}] Fetching details...")
                    
                    detail = self.scrape_detail_page(full_url)
                    
                    if detail['title'] != 'Unknown' and detail['price']:
                        self.vehicles.append(detail)
                        new_vehicles += 1
                        print(f"      ✅ {detail['make']} {detail['model']} | Rs {detail['price']:,} | {detail['engine_cc']}cc")
                    else:
                        print(f"      ⏭️ Skipped - Missing Data. (Title found: {detail['title'] != 'Unknown'}, Price found: {detail['price'] is not None})")
                    
                    delay = random.uniform(2, 4)
                    time.sleep(delay)
                
                print(f"   Added {new_vehicles} bikes from this page")
                return True
            else:
                print(f"   ❌ HTTP {response.status_code}. Possible bot block.")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def scrape_pages(self, num_pages=1):
        for page in range(1, num_pages + 1):
            success = self.scrape_page(page)
            if not success:
                break
            if page < num_pages:
                delay = random.uniform(4, 7)
                print(f"   Waiting {delay:.1f} seconds to avoid rate limits...")
                time.sleep(delay)
        
        return self.vehicles
    
    def save_to_csv(self):
        if not self.vehicles:
            print("❌ No data to save")
            return None
        
        filename = f'ikman_bikes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        fieldnames = ['title', 'price', 'make', 'model', 'yom', 'mileage', 
                     'engine_cc', 'location', 'url', 'scrape_date']
        
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.vehicles)
        
        print(f"\n✅ Saved {len(self.vehicles)} bikes to {filename}")
        return filename

if __name__ == "__main__":
    print("="*60)
    print("🏍️ IKMAN.LK MOTORCYCLE SCRAPER (SIBLING TRAVERSAL FIX)")
    print("="*60)
    
    scraper = IkmanBikeScraper()
    pages = int(input("\nSearch pages to scrape (1-2 recommended): ") or "1")
    
    print(f"\n🔍 Scraping {pages} search page(s)...")
    vehicles = scraper.scrape_pages(num_pages=pages)
    
    if vehicles:
        filename = scraper.save_to_csv()
        print(f"\n📁 Data saved to: {filename}")
        print(f"\n📊 Total bikes: {len(vehicles)}")
    else:
        print("\n❌ No bikes scraped. Check the diagnostic output above.")